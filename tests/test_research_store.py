"""Latest result per symbol: served at once, refreshed in the background, one run at a time."""

import json
import threading
import time

import pytest

from tradingagents.research_store import ResearchStore


class _Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def stores():
    built = []
    yield built
    for store in built:
        assert store.wait_idle(timeout=10)
        store.shutdown()


def _store(stores, **kwargs):
    store = ResearchStore(**{"max_concurrent": 2, **kwargs})
    stores.append(store)
    return store


@pytest.mark.unit
def test_the_first_request_waits_and_later_ones_get_the_stored_result_at_once(stores):
    store = _store(stores)
    runs = []

    def compute():
        runs.append(1)
        return {"n": len(runs)}

    first = store.get("BTC", compute)
    assert (first.value, first.cached, first.refreshing) == ({"n": 1}, False, False)

    second = store.get("BTC", compute)
    assert second.value == {"n": 1}                       # the stored one, not the refresh
    assert second.cached and second.refreshing

    assert store.wait_idle(timeout=5)
    assert store.get("BTC", compute).value == {"n": 2}   # the refresh replaced it


@pytest.mark.unit
def test_requests_during_a_refresh_share_it_and_never_wait(stores):
    store = _store(stores)
    release = threading.Event()
    runs = []

    def compute():
        runs.append(1)
        if len(runs) > 1:
            release.wait(5)
        return len(runs)

    store.get("ETH", compute)                         # first run, stored
    started = time.monotonic()
    served = [store.get("ETH", compute) for _ in range(5)]
    assert time.monotonic() - started < 1             # nobody waited for the refresh
    assert all(s.cached and s.refreshing and s.value == 1 for s in served)

    release.set()
    assert store.wait_idle(timeout=5)
    assert len(runs) == 2                             # five requests, one refresh


@pytest.mark.unit
def test_concurrent_first_requests_share_one_run(stores):
    store = _store(stores)
    started = threading.Event()
    release = threading.Event()
    runs = []

    def compute():
        runs.append(1)
        started.set()
        release.wait(5)
        return "result"

    results = []
    threads = [threading.Thread(target=lambda: results.append(store.get("SOL", compute)))
               for _ in range(3)]
    threads[0].start()
    started.wait(5)
    for t in threads[1:]:
        t.start()
    time.sleep(0.05)
    release.set()
    for t in threads:
        t.join(5)

    assert len(runs) == 1
    assert [r.value for r in results] == ["result"] * 3
    assert not any(r.cached for r in results)


@pytest.mark.unit
def test_a_failed_refresh_keeps_the_previous_result(stores, caplog):
    store = _store(stores)
    store.get("BTC", lambda: "good")

    def failing():
        raise RuntimeError("provider down")

    assert store.get("BTC", failing).value == "good"
    assert store.wait_idle(timeout=5)
    assert store.get("BTC", lambda: "next").value == "good"
    assert "keeping the previous result" in caplog.text


@pytest.mark.unit
def test_a_failed_first_run_reaches_the_caller_and_is_not_stored(stores):
    store = _store(stores)

    def failing():
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        store.get("BTC", failing)
    assert store.peek("BTC") is None
    assert store.get("BTC", lambda: "ok").value == "ok"


@pytest.mark.unit
def test_runs_beyond_the_worker_count_queue(stores):
    store = _store(stores, max_concurrent=1)
    active, peak = [], []
    lock = threading.Lock()

    def compute():
        with lock:
            active.append(1)
            peak.append(len(active))
        time.sleep(0.05)
        with lock:
            active.pop()
        return "done"

    threads = [threading.Thread(target=store.get, args=(f"k{i}", compute)) for i in range(3)]
    [t.start() for t in threads]
    [t.join(5) for t in threads]
    assert max(peak) == 1


@pytest.mark.unit
def test_a_young_result_is_not_refreshed(stores):
    clock = _Clock()
    store = _store(stores, min_refresh_age=60, clock=clock)
    store.get("BTC", lambda: 1)
    assert store.get("BTC", lambda: 2).refreshing is False
    clock.now += 61
    assert store.get("BTC", lambda: 2).refreshing is True


@pytest.mark.unit
def test_refresh_can_be_turned_off_per_lookup(stores):
    store = _store(stores)
    store.get("AAPL@2026-01-05", lambda: "old")
    hit = store.get("AAPL@2026-01-05", lambda: "new", refresh=False)
    assert hit.cached and not hit.refreshing and hit.value == "old"


@pytest.mark.unit
def test_callers_get_independent_copies(stores):
    store = _store(stores, min_refresh_age=10**9)
    first = store.get("k", lambda: {"notices": ["a"]}).value
    first["notices"].append("changed")
    assert store.get("k", lambda: {}).value == {"notices": ["a"]}


@pytest.mark.unit
def test_results_are_saved_and_read_back(stores, tmp_path):
    path = tmp_path / "nested" / "research_results.json"
    store = _store(stores, path=path)
    store.get(("BTC-USD", "crypto", "latest"), lambda: {"score": 47, "label": "Neutral"})

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["version"] == 1
    assert saved["results"][0]["key"] == ["BTC-USD", "crypto", "latest"]

    reloaded = _store(stores, path=path, min_refresh_age=10**9)
    hit = reloaded.get(("BTC-USD", "crypto", "latest"), lambda: pytest.fail("must not recompute"))
    assert hit.cached and hit.value == {"score": 47, "label": "Neutral"}


@pytest.mark.unit
def test_an_unreadable_file_is_ignored(stores, tmp_path):
    path = tmp_path / "research_results.json"
    path.write_text("{not json", encoding="utf-8")
    store = _store(stores, path=path)
    assert store.peek("BTC") is None
    assert store.get("BTC", lambda: "fresh").value == "fresh"


@pytest.mark.unit
def test_the_oldest_results_are_evicted_beyond_the_cap(stores):
    clock = _Clock()
    store = _store(stores, max_entries=2, clock=clock, min_refresh_age=10**9)
    for key in ("a", "b", "c"):
        clock.now += 1
        store.get(key, lambda k=key: k)
    assert store.peek("a") is None
    assert store.peek("b") == "b" and store.peek("c") == "c"


@pytest.mark.unit
def test_the_llm_timeout_is_forwarded_to_clients():
    from tradingagents.llm_clients import build_llm_kwargs

    assert build_llm_kwargs({"llm_provider": "openai", "llm_timeout": 45})["timeout"] == 45.0
    assert "timeout" not in build_llm_kwargs({"llm_provider": "bedrock", "llm_timeout": 45})
    assert "timeout" not in build_llm_kwargs({"llm_provider": "openai", "llm_timeout": None})
    with pytest.raises(ValueError):
        build_llm_kwargs({"llm_provider": "openai", "llm_timeout": -1})
