"""CoinGecko crypto fundamentals: header auth, parsing, and honest placeholders."""

from __future__ import annotations

import logging

import pytest
import requests

from tradingagents.dataflows import evidence
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.vendors import coingecko

KEY = "CG-test-key-not-real"

COIN = {
    "id": "bitcoin",
    "symbol": "btc",
    "name": "Bitcoin",
    "categories": ["Cryptocurrency", "Layer 1 (L1)", ""],
    "genesis_date": "2009-01-03",
    "hashing_algorithm": "SHA-256",
    "market_cap_rank": 1,
    "sentiment_votes_up_percentage": 81.5,
    "market_data": {
        "current_price": {"usd": 64321.5},
        "price_change_percentage_24h": 3.2,
        "price_change_percentage_7d": -4.4,
        "market_cap": {"usd": 1_280_000_000_000},
        "fully_diluted_valuation": {"usd": 1_365_000_000_000},
        "total_volume": {"usd": 32_000_000_000},
        "ath_change_percentage": {"usd": -11.7},
        "circulating_supply": 19_700_000,
        "total_supply": 21_000_000,
        "max_supply": 21_000_000,
    },
    "developer_data": {"commit_count_4_weeks": 0, "stars": 73000, "forks": 36000,
                       "pull_requests_merged": 11000, "pull_request_contributors": 850},
    "community_data": {"reddit_subscribers": 6_500_000, "telegram_channel_user_count": None},
}


class _Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture
def api(monkeypatch):
    """Record CoinGecko requests and answer from ``answers`` (path -> (status, payload))."""
    monkeypatch.setenv("COINGECKO_API_KEY", KEY)
    monkeypatch.delenv("COINGECKO_API_PLAN", raising=False)
    monkeypatch.delenv("COINGECKO_API_BASE_URL", raising=False)
    monkeypatch.setattr(coingecko, "_resolved_ids", {})
    calls = []
    answers = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params or {}, "headers": headers or {}, "timeout": timeout})
        path = url.split("/api/v3/", 1)[1]
        status, payload = answers.get(path, (404, {}))
        if isinstance(status, Exception):
            raise status
        return _Response(status, payload)

    monkeypatch.setattr(coingecko.requests, "get", fake_get)
    return calls, answers


@pytest.mark.unit
def test_the_key_goes_in_the_demo_header_never_the_url(api):
    calls, answers = api
    answers["coins/bitcoin"] = (200, COIN)

    coingecko.fetch_coingecko_fundamentals("BTC-USD", get_current_date())

    call = calls[0]
    assert call["url"] == "https://api.coingecko.com/api/v3/coins/bitcoin"
    assert call["headers"]["x-cg-demo-api-key"] == KEY
    assert KEY not in call["url"] and KEY not in str(call["params"])
    assert call["timeout"] == coingecko.REQUEST_TIMEOUT


@pytest.mark.unit
def test_the_pro_plan_uses_the_pro_root_and_header(api, monkeypatch):
    calls, answers = api
    monkeypatch.setenv("COINGECKO_API_PLAN", "pro")
    answers["coins/bitcoin"] = (200, COIN)

    coingecko.fetch_coingecko_fundamentals("BTC-USD")

    assert calls[0]["url"].startswith("https://pro-api.coingecko.com/api/v3/")
    assert "x-cg-pro-api-key" in calls[0]["headers"]


@pytest.mark.unit
def test_the_block_has_fundamentals_and_no_price_statistics(api):
    _, answers = api
    answers["coins/bitcoin"] = (200, COIN)

    out = coingecko.fetch_coingecko_fundamentals("BTC-USD")

    for expected in ("Market Cap Rank: 1", "Fully Diluted Valuation: $1,365,000,000,000",
                     "FDV / Market Cap: 1.07x", "Circulating Supply: 19,700,000",
                     "Max Supply: 21,000,000", "Circulating / Max (or Total): 93.8%",
                     "Commits (last 4 weeks): 0", "Reddit Subscribers: 6,500,000",
                     "Below All-Time High: 11.7%", "Genesis Date: 2009-01-03"):
        assert expected in out, expected
    for price_stat in ("64321", "64,321", "3.2", "-4.4", "Telegram"):
        assert price_stat not in out


@pytest.mark.unit
def test_a_served_coin_is_recorded_as_evidence(api):
    _, answers = api
    answers["coins/bitcoin"] = (200, COIN)

    with evidence.collecting() as collector, evidence.analyst_scope("fundamentals"):
        coingecko.fetch_coingecko_fundamentals("BTC-USD")

    [item] = collector.items_for("fundamentals")
    assert item["url"] == "https://www.coingecko.com/en/coins/bitcoin"
    assert item["source"] == "CoinGecko" and item["type"] == "data"


@pytest.mark.unit
def test_an_unknown_base_is_resolved_by_search_once(api):
    calls, answers = api
    answers["search"] = (200, {"coins": [{"id": "fake-coin-imposter", "symbol": "ZZZX2"},
                                         {"id": "zzzx-network", "symbol": "ZZZX"}]})
    answers["coins/zzzx-network"] = (200, {**COIN, "id": "zzzx-network", "name": "ZZZX"})

    first = coingecko.fetch_coingecko_fundamentals("ZZZX-USD")
    coingecko.fetch_coingecko_fundamentals("ZZZX-USD")

    assert "ZZZX" in first
    assert [c["url"].rsplit("/", 1)[-1] for c in calls].count("search") == 1


@pytest.mark.unit
def test_a_base_with_no_match_is_a_placeholder(api):
    _, answers = api
    answers["search"] = (200, {"coins": []})
    out = coingecko.fetch_coingecko_fundamentals("NOPE-USD")
    assert coingecko.is_unavailable(out) and "NOPE" in out


@pytest.mark.unit
def test_a_missing_key_is_a_placeholder_without_a_request(api, monkeypatch):
    calls, _ = api
    monkeypatch.delenv("COINGECKO_API_KEY")
    out = coingecko.fetch_coingecko_fundamentals("BTC-USD")
    assert coingecko.is_unavailable(out) and "COINGECKO_API_KEY" in out
    assert calls == []


@pytest.mark.unit
@pytest.mark.parametrize("status, expected", [
    (401, "rejected"), (403, "rejected"), (429, "rate limited"), (500, "HTTP 500"),
])
def test_http_errors_are_placeholders(api, status, expected):
    _, answers = api
    answers["coins/bitcoin"] = (status, {})
    out = coingecko.fetch_coingecko_fundamentals("BTC-USD")
    assert coingecko.is_unavailable(out) and expected in out


@pytest.mark.unit
def test_network_errors_and_bad_json_are_placeholders_without_the_key(api, caplog):
    _, answers = api
    answers["coins/bitcoin"] = (requests.ConnectionError(f"boom key={KEY}"), None)
    with caplog.at_level(logging.WARNING):
        out = coingecko.fetch_coingecko_fundamentals("BTC-USD")
    assert coingecko.is_unavailable(out) and KEY not in out
    assert KEY not in caplog.text

    answers["coins/bitcoin"] = (200, ValueError("not json"))
    assert "invalid JSON" in coingecko.fetch_coingecko_fundamentals("BTC-USD")


@pytest.mark.unit
def test_a_historical_run_withholds_the_live_snapshot(api):
    calls, _ = api
    out = coingecko.fetch_coingecko_fundamentals("BTC-USD", "2024-01-05")
    assert coingecko.is_unavailable(out) and "withheld" in out
    assert calls == []


@pytest.mark.unit
@pytest.mark.parametrize("ticker, base", [("BTC-USD", "BTC"), ("SHIB-USD", "SHIB"), ("pepe", "PEPE")])
def test_coin_base(ticker, base):
    assert coingecko.coin_base(ticker) == base
