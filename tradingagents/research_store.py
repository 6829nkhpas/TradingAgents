"""The latest research result for every symbol, refreshed in the background.

- Every symbol keeps its last computed result. A request for a symbol that has
  one gets that result straight away.
- That request also queues a fresh run for the symbol, unless one is already
  queued or running for it. When the run finishes it replaces the stored result,
  so the next request gets the newer one. Several requests for the same symbol
  therefore share a single run, and none of them waits for it.
- A symbol with no result yet has nothing to serve: the request waits for the
  first run, and concurrent requests for that symbol wait on the same run.
- Runs go through a queue served by ``max_concurrent`` workers, so at most that
  many execute at once.
- A failed refresh keeps the previous result. A failed first run is reported to
  everyone waiting on it and nothing is stored, so the next request tries again.
- With a ``path`` the results are written to a JSON file after every run and
  read back at startup, so a restart or redeploy keeps them.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections.abc import Callable, Hashable
from concurrent.futures import Future, ThreadPoolExecutor, wait as wait_futures
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tradingagents.dataflows.files import replace_file

logger = logging.getLogger(__name__)

_FILE_VERSION = 1


@dataclass
class _Entry:
    value: Any
    stored_at: float


@dataclass
class Lookup:
    """A served result and where it came from."""

    value: Any
    cached: bool       # served from a run made before this request
    refreshing: bool   # a newer run for the key is queued or running


class ResearchStore:
    def __init__(
        self,
        max_concurrent: int,
        *,
        path: str | Path | None = None,
        min_refresh_age: float = 0.0,
        max_entries: int = 1000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.min_refresh_age = max(0.0, float(min_refresh_age))
        self.max_entries = max(1, int(max_entries))
        self._path = Path(path) if path else None
        self._clock = clock
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._entries: dict[Hashable, _Entry] = {}
        self._inflight: dict[Hashable, Future] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_concurrent)), thread_name_prefix="research-run"
        )
        if self._path is not None:
            self._load()

    # -- lookups -------------------------------------------------------------

    def get(self, key: Hashable, compute: Callable[[], Any], *, refresh: bool = True) -> Lookup:
        """The stored result for ``key`` (queuing a refresh), else the first run's.

        ``refresh=False`` serves a stored result without queuing a new run, for
        results that cannot change (a historical analysis date).
        """
        with self._lock:
            entry = self._entries.get(key)
            future = self._inflight.get(key)
            if entry is not None:
                stale = self._clock() - entry.stored_at >= self.min_refresh_age
                if refresh and future is None and stale:
                    future = self._submit(key, compute)
                return Lookup(copy.deepcopy(entry.value), cached=True, refreshing=future is not None)
            if future is None:
                future = self._submit(key, compute)

        value = future.result()  # re-raises the run's failure
        return Lookup(copy.deepcopy(value), cached=False, refreshing=False)

    def peek(self, key: Hashable) -> Any | None:
        """The stored result for ``key`` without queuing anything; None when there is none."""
        with self._lock:
            entry = self._entries.get(key)
            return copy.deepcopy(entry.value) if entry is not None else None

    def is_refreshing(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._inflight

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until no run is queued or running; False if ``timeout`` ran out."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                pending = list(self._inflight.values())
            if not pending:
                return True
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            wait_futures(pending, timeout=remaining)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- runs ----------------------------------------------------------------

    def _submit(self, key: Hashable, compute: Callable[[], Any]) -> Future:
        """Queue a run for ``key``; the caller holds ``self._lock``."""
        future = self._executor.submit(self._run, key, compute)
        self._inflight[key] = future
        return future

    def _run(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        try:
            value = compute()
        except BaseException as exc:
            with self._lock:
                self._inflight.pop(key, None)
                had_result = key in self._entries
            if had_result:
                # Nobody waits on a refresh, so its failure would otherwise vanish.
                logger.warning("Research refresh failed for %s; keeping the previous result: %s",
                               key, exc)
            raise

        with self._lock:
            self._entries[key] = _Entry(value, self._clock())
            self._evict()
            self._inflight.pop(key, None)
            snapshot = [(k, e.stored_at, e.value) for k, e in self._entries.items()]
        if self._path is not None:
            self._save(snapshot)
        return value

    def _evict(self) -> None:
        """Drop the oldest results beyond ``max_entries``; the caller holds the lock."""
        excess = len(self._entries) - self.max_entries
        if excess <= 0:
            return
        oldest = sorted(
            (k for k in self._entries if k not in self._inflight),
            key=lambda k: self._entries[k].stored_at,
        )
        for key in oldest[:excess]:
            del self._entries[key]

    # -- persistence -----------------------------------------------------------

    def _save(self, snapshot: list[tuple[Hashable, float, Any]]) -> None:
        payload = {
            "version": _FILE_VERSION,
            "results": [
                {"key": list(key) if isinstance(key, tuple) else key, "stored_at": stored_at,
                 "result": value}
                for key, stored_at, value in snapshot
            ],
        }

        def write(temp: str) -> None:
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, default=str)

        with self._save_lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                replace_file(self._path, write)
            except OSError as exc:
                logger.warning("Could not save research results to %s: %s", self._path, exc)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            rows = payload.get("results", []) if payload.get("version") == _FILE_VERSION else []
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning("Ignoring unreadable research results file %s: %s", self._path, exc)
            return
        for row in rows:
            try:
                key = tuple(row["key"]) if isinstance(row["key"], list) else row["key"]
                self._entries[key] = _Entry(row["result"], float(row["stored_at"]))
            except (KeyError, TypeError, ValueError):
                continue
        self._evict()
        logger.info("Loaded %d stored research results from %s", len(self._entries), self._path)
