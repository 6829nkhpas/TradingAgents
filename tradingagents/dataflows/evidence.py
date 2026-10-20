"""Source evidence recorded during a research run.

The vendors already format their data as text for the analysts' prompts. Next
to that, they record each article, post, market or data page they served here,
so the API can return real links for every analyst without asking a model to
repeat (or invent) URLs.

A run opens a collector with :func:`collecting`; each analyst runs inside
:func:`analyst_scope`, which tags what its data calls record. Both are
``ContextVar``s, the same mechanism ``dataflows.config.run_config`` uses, and
LangGraph carries the caller's context into its node and tool threads, so the
tagging holds for parallel analysts and their tool calls. Outside a run every
call here is a no-op, so vendors can record unconditionally.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

EVIDENCE_TYPES = ("article", "social_post", "prediction_market", "data")
DEFAULT_CAP = 15
_TITLE_CHARS = 200

_collector: ContextVar[EvidenceCollector | None] = ContextVar(
    "tradingagents_evidence_collector", default=None
)
_current_analyst: ContextVar[str | None] = ContextVar(
    "tradingagents_evidence_analyst", default=None
)


def _iso(value: Any) -> str | None:
    """A publish time as an ISO 8601 string, or None when unusable."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=UTC).isoformat().replace("+00:00", "Z")
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return _iso(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        return None


def _clean(text: Any, limit: int) -> str:
    """Single-line, trimmed text, cut to ``limit`` characters."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


class EvidenceCollector:
    """Thread-safe store for one run's evidence items and notices."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, list[dict]] = {}
        self._notices: list[str] = []

    def add(self, analyst: str, item: dict) -> None:
        with self._lock:
            self._items.setdefault(analyst, []).append(item)

    def add_notice(self, text: str) -> None:
        with self._lock:
            if text not in self._notices:
                self._notices.append(text)

    @property
    def notices(self) -> list[str]:
        with self._lock:
            return list(self._notices)

    def items_for(self, analyst: str, cap: int = DEFAULT_CAP) -> list[dict]:
        """The analyst's evidence: deduplicated, newest first, capped, with ids."""
        with self._lock:
            raw = list(self._items.get(analyst, []))

        seen: set[tuple] = set()
        unique: list[dict] = []
        for item in raw:
            key = (item["url"],) if item.get("url") else (item["title"].lower(), item["source"].lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)

        # Newest first; undated items keep their order after the dated ones.
        dated = sorted((i for i in unique if i.get("published_at")),
                       key=lambda i: i["published_at"], reverse=True)
        undated = [i for i in unique if not i.get("published_at")]
        ordered = (dated + undated)[:cap]
        return [{"id": f"e{n}", **item} for n, item in enumerate(ordered, start=1)]


@contextmanager
def collecting() -> Iterator[EvidenceCollector]:
    """Collect evidence for the length of the block (one research run)."""
    collector = EvidenceCollector()
    token = _collector.set(collector)
    try:
        yield collector
    finally:
        _collector.reset(token)


@contextmanager
def analyst_scope(name: str) -> Iterator[None]:
    """Tag evidence recorded inside the block as served to analyst ``name``."""
    token = _current_analyst.set(name)
    try:
        yield
    finally:
        _current_analyst.reset(token)


def current_analyst() -> str | None:
    return _current_analyst.get()


def record(
    *,
    type: str,
    title: Any,
    source: Any,
    url: Any = None,
    published_at: Any = None,
    stance: Any = None,
) -> None:
    """Record one served item for the analyst in scope; a no-op outside a run.

    An item needs a title and a source. ``url`` is kept only when it is a real
    http(s) link from the vendor's own data.
    """
    collector = _collector.get()
    analyst = _current_analyst.get()
    if collector is None or analyst is None:
        return
    if type not in EVIDENCE_TYPES:
        raise ValueError(f"unknown evidence type: {type!r}")
    title_text = _clean(title, _TITLE_CHARS)
    source_text = _clean(source, 80)
    if not title_text or not source_text:
        return

    item: dict[str, str] = {"type": type, "title": title_text, "source": source_text}
    link = str(url or "").strip()
    if link.startswith(("https://", "http://")):
        item["url"] = link
    published = _iso(published_at)
    if published:
        item["published_at"] = published
    stance_text = _clean(stance, 40)
    if stance_text:
        item["stance"] = stance_text
    collector.add(analyst, item)


def add_notice(text: str) -> None:
    """Add a run-level notice for the response; a no-op outside a run."""
    collector = _collector.get()
    if collector is not None and text:
        collector.add_notice(text)
