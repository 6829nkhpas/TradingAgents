"""Research API: graph shape, convictions, overall conviction and the HTTP contract."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

import tradingagents.research as research
from tradingagents.agents.analysts import (
    crypto_fundamentals_analyst as crypto_fund,
    sentiment_analyst as sentiment,
)
from tradingagents.agents.schemas import AnalystConviction, SentimentBand, SentimentReport
from tradingagents.dataflows import evidence
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import normalize_crypto_symbol
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.setup import CRYPTO_ANALYSTS, STOCK_ANALYSTS, GraphSetup
from tradingagents.research import (
    InvalidResearchRequest,
    ResearchAnalyzer,
    ResearchUnavailable,
    _canonical_analysis_date,
    configured_research_sources,
    normalize_research_instrument,
)

SCORES = {"sentiment": 66, "news": 61, "fundamentals": 44}


def _text(messages) -> str:
    if hasattr(messages, "to_messages"):
        messages = messages.to_messages()
    if isinstance(messages, str):
        return messages
    return "\n".join(
        m.content if isinstance(m, BaseMessage) and isinstance(m.content, str) else str(m)
        for m in messages
    )


def _section(text: str) -> str:
    """Which analyst a prompt (or a report being scored) belongs to."""
    if "[sentiment]" in text or "sentiment analyst" in text:
        return "sentiment"
    if "[news]" in text or "news researcher" in text:
        return "news"
    return "fundamentals"


class FakeLLM:
    """A chat model that writes one report per analyst and scores it.

    ``fail`` names analysts whose report call raises; ``unscorable`` makes the
    conviction call fail for every report.
    """

    def __init__(self, fail=(), unscorable=False):
        self.fail = set(fail)
        self.unscorable = unscorable

    def _report(self, messages) -> AIMessage:
        name = _section(_text(messages))
        if name in self.fail:
            raise RuntimeError(f"{name} provider error")
        return AIMessage(content=f"[{name}] report body with a summary table.")

    def bind_tools(self, tools):
        return RunnableLambda(self._report)

    def invoke(self, messages, *args, **kwargs):
        return self._report(messages)

    def with_structured_output(self, schema, **kwargs):
        if schema is SentimentReport:
            def write(messages):
                if "sentiment" in self.fail:
                    raise RuntimeError("sentiment provider error")
                return SentimentReport(overall_band=SentimentBand.BULLISH, overall_score=6.6,
                                       confidence="high", narrative="[sentiment] narrative.")
            return RunnableLambda(write)
        if schema is AnalystConviction:
            def score(messages):
                if self.unscorable:
                    raise ValueError("bad JSON")
                name = _section(_text(messages))
                return AnalystConviction(
                    score=SCORES[name], confidence=0.6, thesis=f"{name} thesis",
                    drivers=[{"headline": f"{name} driver", "impact": "positive", "weight": 0.4}],
                    risks=[f"{name} risk"], horizon="short-term",
                )
            return RunnableLambda(score)
        raise NotImplementedError(schema)


@pytest.fixture
def offline_sources(monkeypatch):
    """Stub every pre-fetched source; each records one evidence item like the real ones."""
    def social(name):
        def fetch(ticker, *a, **k):
            evidence.record(type="social_post", title=f"{name} post about {ticker}", source=name,
                            url=f"https://example.com/{name}/1", published_at="2026-09-01T10:00:00Z")
            return f"{name} block"
        return fetch

    monkeypatch.setattr(sentiment, "fetch_stocktwits_messages", social("StockTwits"))
    monkeypatch.setattr(sentiment, "fetch_reddit_posts", social("Reddit"))
    monkeypatch.setattr(sentiment.get_news, "func", lambda *a, **k: "news block", raising=False)
    monkeypatch.setattr(crypto_fund, "fetch_coingecko_fundamentals", lambda *a, **k: "coingecko block")
    monkeypatch.setattr(crypto_fund, "_yahoo_block", lambda *a, **k: "yahoo block")
    monkeypatch.setattr(research, "resolve_instrument_identity", lambda ticker: {})


@pytest.fixture
def analyzers():
    """Analyzers a test builds; their background runs finish before teardown undoes its stubs."""
    built = []
    yield built
    for analyzer in built:
        assert analyzer.store.wait_idle(timeout=30)
        analyzer.store.shutdown()


def _analyzer(monkeypatch, llm, built=None, **config):
    class _Client:
        def get_llm(self):
            return llm

    monkeypatch.setattr(research, "create_llm_client", lambda **kwargs: _Client())
    # No file, and no background refresh unless a test asks for one.
    settings = {"research_persist_results": False, "research_refresh_min_age_seconds": 10**9, **config}
    analyzer = ResearchAnalyzer(settings)
    if built is not None:
        built.append(analyzer)
    return analyzer


def _nulls(value, path="$"):
    if value is None:
        return [path]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _nulls(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in _nulls(v, f"{path}[{i}]")]
    return []


# --- graph shape -------------------------------------------------------------


@pytest.mark.unit
def test_research_graph_contains_the_three_analysts_only():
    """No market/technical analyst, and no debate, trading, risk or portfolio nodes."""
    setup = GraphSetup(FakeLLM(), max_tool_rounds=1)
    stock = setup.setup_research_graph(STOCK_ANALYSTS)
    crypto = setup.setup_research_graph(CRYPTO_ANALYSTS)

    expected = {"Sentiment Analyst", "News Analyst", "Fundamentals Analyst"}
    assert set(stock.nodes) == expected
    assert set(crypto.nodes) == expected


@pytest.mark.unit
def test_the_market_analyst_cannot_be_selected():
    with pytest.raises(ValueError, match="unknown analyst key"):
        GraphSetup(FakeLLM(), max_tool_rounds=1).setup_research_graph(("market", "social"))


# --- instrument and date handling -------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("symbol", "asset_type", "expected"),
    [
        ("AAPL", None, ("AAPL", "stock")),
        ("btc-usd", None, ("BTC-USD", "crypto")),
        ("Bitcoin", None, ("BTC-USD", "crypto")),
        ("BTC", "crypto", ("BTC-USD", "crypto")),
        ("SHIB", "crypto", ("SHIB-USD", "crypto")),
        ("BTC", "stock", ("BTC", "stock")),
    ],
)
def test_normalize_research_instrument(symbol, asset_type, expected):
    assert normalize_research_instrument(symbol, asset_type) == expected


@pytest.mark.unit
def test_crypto_symbol_normalization_accepts_unlisted_bases():
    assert normalize_crypto_symbol("SHIB/USDT") == "SHIB-USD"


@pytest.mark.unit
def test_invalid_or_unsafe_symbols_are_rejected():
    with pytest.raises(InvalidResearchRequest):
        normalize_research_instrument("../../etc/passwd")
    with pytest.raises(InvalidResearchRequest):
        normalize_research_instrument("   ")
    with pytest.raises(InvalidResearchRequest):
        normalize_research_instrument("AAPL", "forex")


@pytest.mark.unit
def test_analysis_date_defaults_to_today_and_rejects_future():
    assert _canonical_analysis_date(None) == get_current_date()
    future = (date.fromisoformat(get_current_date()) + timedelta(days=1)).isoformat()
    with pytest.raises(InvalidResearchRequest, match="future"):
        _canonical_analysis_date(future)


@pytest.mark.unit
def test_configured_sources_name_the_platforms_used_by_the_analysts():
    stock = configured_research_sources(DEFAULT_CONFIG, asset_type="stock")
    assert set(stock) == {"sentiment", "news", "fundamentals"}
    assert "StockTwits" in stock["sentiment"] and "Reddit" in stock["sentiment"]
    assert "SEC EDGAR" in stock["fundamentals"]
    assert "FRED" in stock["news"] and "Polymarket" in stock["news"]

    crypto = configured_research_sources(DEFAULT_CONFIG, asset_type="crypto")
    assert crypto["fundamentals"] == ["CoinGecko", "Yahoo Finance"]


@pytest.mark.unit
def test_crypto_runs_search_crypto_macro_news(monkeypatch, offline_sources):
    analyzer = _analyzer(monkeypatch, FakeLLM())
    crypto = analyzer._run_config("crypto")
    stock = analyzer._run_config("stock")
    assert crypto["global_news_queries"] == DEFAULT_CONFIG["crypto_global_news_queries"]
    assert stock["global_news_queries"] == DEFAULT_CONFIG["global_news_queries"]


# --- convictions -------------------------------------------------------------


@pytest.mark.unit
def test_a_stock_run_returns_three_convictions_and_an_overall(monkeypatch, offline_sources):
    result = _analyzer(monkeypatch, FakeLLM()).analyze("AAPL")

    assert result["symbol"] == "AAPL" and result["asset_type"] == "stock"
    assert "base_asset" not in result
    assert set(result["analysts"]) == {"sentiment", "news", "fundamentals"}
    for name, analyst in result["analysts"].items():
        assert analyst["status"] == "ok", analyst
        assert analyst["conviction"]["score"] == SCORES[name]
        assert analyst["report"].startswith(("[", "**Overall Sentiment"))
        assert analyst["sources"]
    assert result["analysts"]["sentiment"]["conviction"]["label"] == "Bullish"
    assert result["analysts"]["fundamentals"]["conviction"]["label"] == "Neutral"

    overall = result["overall"]
    assert overall["components"] == ["sentiment", "news", "fundamentals"]
    assert 44 < overall["score"] < 66
    assert result["cached"] is False
    assert "market" not in result and "market" not in result["analysts"]
    assert "research" not in result and "configured_sources" not in result
    assert "technical" not in json.dumps(result, default=str).lower()


@pytest.mark.unit
def test_social_evidence_is_tagged_to_the_sentiment_analyst(monkeypatch, offline_sources):
    result = _analyzer(monkeypatch, FakeLLM()).analyze("AAPL")
    items = result["analysts"]["sentiment"]["evidence"]
    assert {i["source"] for i in items} == {"StockTwits", "Reddit"}
    assert [i["id"] for i in items] == ["e1", "e2"]
    assert "evidence" not in result["analysts"]["news"]


@pytest.mark.unit
def test_a_crypto_run_gets_all_three_convictions(monkeypatch, offline_sources):
    result = _analyzer(monkeypatch, FakeLLM()).analyze("BTC", asset_type="crypto")

    assert result["symbol"] == "BTC-USD" and result["base_asset"] == "BTC"
    assert set(result["analysts"]) == {"sentiment", "news", "fundamentals"}
    assert all(a["status"] == "ok" for a in result["analysts"].values())
    assert result["analysts"]["fundamentals"]["sources"] == ["CoinGecko", "Yahoo Finance"]
    assert research.CRYPTO_NOTICE in result["notices"]


@pytest.mark.unit
def test_an_unscorable_report_is_degraded_and_has_no_overall(monkeypatch, offline_sources):
    result = _analyzer(monkeypatch, FakeLLM(unscorable=True)).analyze("AAPL")

    for analyst in result["analysts"].values():
        assert analyst["status"] == "degraded"
        assert "conviction" not in analyst
        assert analyst["report"]
    assert "overall" not in result
    assert research.NO_OVERALL_NOTICE in result["notices"]


@pytest.mark.unit
def test_a_failing_analyst_is_unavailable_and_the_rest_still_report(monkeypatch, offline_sources):
    result = _analyzer(monkeypatch, FakeLLM(fail={"news"})).analyze("AAPL")

    news = result["analysts"]["news"]
    assert news["status"] == "unavailable"
    assert "News Analyst failed" in news["reason"]
    assert "report" not in news and "conviction" not in news
    assert result["analysts"]["sentiment"]["status"] == "ok"
    assert result["overall"]["components"] == ["sentiment", "fundamentals"]


@pytest.mark.unit
def test_every_analyst_failing_is_an_error(monkeypatch, offline_sources):
    analyzer = _analyzer(monkeypatch, FakeLLM(fail={"sentiment", "news", "fundamentals"}))
    with pytest.raises(ResearchUnavailable, match="No analyst produced a report"):
        analyzer.analyze("AAPL")


@pytest.mark.unit
def test_a_repeated_request_is_served_the_stored_result_and_refreshes_it(
        monkeypatch, offline_sources, analyzers):
    analyzer = _analyzer(monkeypatch, FakeLLM(), analyzers, research_refresh_min_age_seconds=0)
    calls = []
    real = analyzer._run
    monkeypatch.setattr(analyzer, "_run", lambda *a: calls.append(a) or real(*a))

    first = analyzer.analyze("BTC", asset_type="crypto")
    second = analyzer.analyze("bitcoin")      # same instrument, another spelling

    assert first["cached"] is False and first["refreshing"] is False
    assert second["cached"] is True and second["refreshing"] is True
    assert second["generated_at"] == first["generated_at"]   # the stored result, at once

    assert analyzer.store.wait_idle(timeout=30)
    assert len(calls) == 2                                  # the refresh ran in the background
    third = analyzer.analyze("BTC-USD")
    assert third["cached"] is True
    assert third["generated_at"] >= first["generated_at"]


@pytest.mark.unit
def test_a_past_date_is_computed_once_and_never_refreshed(monkeypatch, offline_sources, analyzers):
    analyzer = _analyzer(monkeypatch, FakeLLM(), analyzers, research_refresh_min_age_seconds=0)
    first = analyzer.analyze("AAPL", analysis_date="2026-01-05")
    again = analyzer.analyze("AAPL", analysis_date="2026-01-05")
    assert first["analysis_date"] == "2026-01-05"
    assert again["cached"] is True and again["refreshing"] is False


@pytest.mark.unit
def test_results_survive_a_restart(monkeypatch, offline_sources, analyzers, tmp_path):
    path = str(tmp_path / "research_results.json")
    first = _analyzer(monkeypatch, FakeLLM(), analyzers, research_persist_results=True,
                      research_store_path=path).analyze("AAPL")

    restarted = _analyzer(monkeypatch, FakeLLM(fail={"sentiment", "news", "fundamentals"}), analyzers,
                          research_persist_results=True, research_store_path=path)
    served = restarted.analyze("AAPL")
    assert served["cached"] is True
    assert served["generated_at"] == first["generated_at"]
    assert served["overall"] == first["overall"]


# --- HTTP contract -----------------------------------------------------------


def _client_with(analyzer):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tradingagents.api.app import app, get_research_analyzer

    app.dependency_overrides[get_research_analyzer] = lambda: analyzer
    return TestClient(app), app


@pytest.mark.unit
def test_the_response_has_no_nulls_and_no_market_section(monkeypatch, offline_sources):
    client, app = _client_with(_analyzer(monkeypatch, FakeLLM(fail={"news"})))
    try:
        with client:
            response = client.post("/api/v1/research", json={"symbol": "BTC", "asset_type": "crypto"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert _nulls(body) == []
        assert "market" not in body["analysts"]
        assert body["analysts"]["news"] == {
            "status": "unavailable",
            "reason": body["analysts"]["news"]["reason"],
            "sources": body["analysts"]["news"]["sources"],
        }
        assert body["overall"]["label"] in ("Bullish", "Bearish", "Neutral")
    finally:
        app.dependency_overrides.clear()


@pytest.mark.unit
def test_the_endpoint_needs_no_key_unless_one_is_configured(monkeypatch, offline_sources):
    client, app = _client_with(_analyzer(monkeypatch, FakeLLM()))
    try:
        with client:
            assert client.get("/health").json()["status"] == "ok"
            assert client.post("/api/v1/research", json={"symbol": "AAPL"}).status_code == 200

            monkeypatch.setenv("TRADINGAGENTS_API_KEY", "test-secret")
            assert client.post("/api/v1/research", json={"symbol": "AAPL"}).status_code == 401
            ok = client.post("/api/v1/research", json={"symbol": "AAPL"},
                             headers={"Authorization": "Bearer test-secret"})
            assert ok.status_code == 200
    finally:
        app.dependency_overrides.clear()


@pytest.mark.unit
def test_bad_input_is_422_and_total_failure_is_502(monkeypatch, offline_sources):
    analyzer = _analyzer(monkeypatch, FakeLLM(fail={"sentiment", "news", "fundamentals"}))
    client, app = _client_with(analyzer)
    try:
        with client:
            assert client.post("/api/v1/research", json={"symbol": "../etc"}).status_code == 422
            assert client.post("/api/v1/research", json={"symbol": "AAPL", "x": 1}).status_code == 422
            failed = client.post("/api/v1/research", json={"symbol": "AAPL"})
            assert failed.status_code == 502
            assert "No analyst produced a report" in failed.json()["detail"]
    finally:
        app.dependency_overrides.clear()
