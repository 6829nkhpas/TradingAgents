"""Research-only API graph, ticker handling and response tests."""

from datetime import date, timedelta

import pytest

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import normalize_crypto_symbol
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.setup import GraphSetup
from tradingagents.research import (
    InvalidResearchRequest,
    _canonical_analysis_date,
    configured_research_sources,
    normalize_research_instrument,
)


@pytest.mark.unit
def test_research_graph_contains_analysts_only():
    """The API graph must not schedule debate, trading, risk or portfolio nodes."""
    workflow = GraphSetup(
        quick_thinking_llm=object(),
        deep_thinking_llm=object(),
        conditional_logic=None,
        max_tool_rounds=1,
    ).setup_research_graph(("market", "social", "news"))

    assert set(workflow.nodes) == {"Market Analyst", "Sentiment Analyst", "News Analyst"}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("symbol", "asset_type", "expected"),
    [
        ("AAPL", None, ("AAPL", "stock")),
        ("btc-usd", None, ("BTC-USD", "crypto")),
        ("Bitcoin", None, ("BTC-USD", "crypto")),
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
    sources = configured_research_sources(DEFAULT_CONFIG, asset_type="stock")
    assert "Yahoo Finance" in sources["market"]
    assert "StockTwits" in sources["sentiment"]
    assert "Reddit" in sources["sentiment"]
    assert "SEC EDGAR" in sources["fundamentals"]
    assert configured_research_sources(DEFAULT_CONFIG, asset_type="crypto")["fundamentals"] == []


@pytest.mark.unit
def test_research_endpoint_contract_when_api_extras_are_installed(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    from fastapi.testclient import TestClient

    from tradingagents.api.app import app, get_research_analyzer

    class StubAnalyzer:
        def analyze(self, symbol, **kwargs):
            return {
                "symbol": symbol,
                "asset_type": kwargs.get("asset_type") or "stock",
                "analysis_date": "2026-10-01",
                "generated_at": "2026-10-01T12:00:00Z",
                "research": {
                    "market": "Technical report",
                    "sentiment": "Sentiment report",
                    "news": "News report",
                    "fundamentals": "Fundamentals report",
                },
                "configured_sources": {
                    "market": ["Yahoo Finance"],
                    "sentiment": ["StockTwits", "Reddit"],
                    "news": ["Yahoo Finance"],
                    "fundamentals": ["SEC EDGAR"],
                },
                "notices": [],
                "disclaimer": "Research only.",
            }

    monkeypatch.setenv("TRADINGAGENTS_API_KEY", "test-secret")
    app.dependency_overrides[get_research_analyzer] = lambda: StubAnalyzer()
    try:
        with TestClient(app) as client:
            assert client.get("/health").json()["status"] == "ok"
            unauthorized = client.post("/api/v1/research", json={"symbol": "AAPL"})
            assert unauthorized.status_code == 401

            response = client.post(
                "/api/v1/research",
                json={"symbol": "AAPL"},
                headers={"Authorization": "Bearer test-secret"},
            )
            assert response.status_code == 200
            result = response.json()
            assert result["research"]["market"] == "Technical report"
            assert result["configured_sources"]["fundamentals"] == ["SEC EDGAR"]
    finally:
        app.dependency_overrides.clear()
