"""Crypto fundamentals analyst, Yahoo's crypto profile, and crypto-aware social sources."""

from __future__ import annotations

from unittest import mock

import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents.analysts import crypto_fundamentals_analyst as analyst
from tradingagents.dataflows import evidence
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import crypto_base
from tradingagents.dataflows.vendors import reddit, stocktwits
from tradingagents.dataflows.vendors.yahoo import fundamentals as yahoo_fundamentals


def _state():
    return {"company_of_interest": "BTC-USD", "asset_type": "crypto",
            "trade_date": get_current_date(), "messages": []}


def _llm(captured):
    llm = mock.MagicMock()
    llm.invoke.side_effect = lambda messages: captured.setdefault("m", messages) and AIMessage(
        content=[{"type": "reasoning", "text": "hidden"}, {"type": "text", "text": "Crypto report."}]
    )
    return llm


@pytest.mark.unit
def test_the_analyst_writes_one_report_from_both_profiles(monkeypatch):
    monkeypatch.setattr(analyst, "fetch_coingecko_fundamentals", lambda *a: "CG BLOCK")
    monkeypatch.setattr(analyst, "_yahoo_block", lambda *a: "YF BLOCK")
    captured = {}

    out = analyst.create_crypto_fundamentals_analyst(_llm(captured))(_state())

    assert out["fundamentals_report"] == "Crypto report."   # text blocks only
    prompt = "\n".join(m.content for m in captured["m"])
    assert "CG BLOCK" in prompt and "YF BLOCK" in prompt
    assert "no technical or chart analysis" in prompt
    assert "do not discuss earnings" in prompt


@pytest.mark.unit
def test_an_unavailable_coingecko_adds_a_yahoo_only_notice(monkeypatch):
    monkeypatch.setattr(analyst, "fetch_coingecko_fundamentals",
                        lambda *a: "<CoinGecko unavailable: rate limited (HTTP 429)>")
    monkeypatch.setattr(analyst, "_yahoo_block", lambda *a: "YF BLOCK")

    with evidence.collecting() as collector:
        analyst.create_crypto_fundamentals_analyst(_llm({}))(_state())

    assert collector.notices == [analyst.COINGECKO_UNAVAILABLE_NOTICE]


@pytest.mark.unit
def test_a_yahoo_failure_becomes_a_placeholder(monkeypatch):
    from tradingagents.dataflows.errors import VendorUnavailableError

    def down(*a):
        raise VendorUnavailableError("Yahoo Finance request failed: TimeoutError")

    monkeypatch.setattr(analyst, "get_crypto_profile", down)
    assert analyst._yahoo_block("BTC-USD", get_current_date()).startswith(
        "<Yahoo Finance crypto profile unavailable")


@pytest.mark.unit
def test_yahoo_crypto_profile_has_supply_and_no_price_statistics(monkeypatch):
    info = {"shortName": "Bitcoin USD", "quoteType": "CRYPTOCURRENCY", "marketCap": 1_280_000_000_000,
            "circulatingSupply": 19_700_000, "volume24Hr": 32_000_000_000, "startDate": 1278979200,
            "fiftyTwoWeekHigh": 73000.0, "fiftyDayAverage": 61000.0, "regularMarketPrice": 65000.0,
            "description": "Bitcoin is a decentralized digital currency."}
    monkeypatch.setattr(yahoo_fundamentals, "yf_retry", lambda fn: info)

    out = yahoo_fundamentals.get_crypto_profile("BTC-USD", get_current_date())

    assert "Circulating Supply: 19700000" in out and "Trading Since: 2010-07-13" in out
    for price_stat in ("73000", "61000", "65000"):
        assert price_stat not in out


@pytest.mark.unit
def test_yahoo_crypto_profile_is_withheld_from_a_historical_run(monkeypatch):
    monkeypatch.setattr(yahoo_fundamentals, "yf_retry", mock.Mock(side_effect=AssertionError("fetched")))
    assert "withheld" in yahoo_fundamentals.get_crypto_profile("BTC-USD", "2024-01-05")


# --- long-tail coins reach the social sources by their base -----------------


@pytest.mark.unit
def test_crypto_base_accepts_any_base_only_when_asked():
    assert crypto_base("SHIB-USD") is None
    assert crypto_base("SHIB-USD", any_base=True) == "SHIB"
    assert crypto_base("BTC-USD") == "BTC"
    assert crypto_base("AAPL", any_base=True) is None


@pytest.mark.unit
def test_stocktwits_maps_a_long_tail_coin_when_it_is_crypto():
    assert stocktwits._stocktwits_symbol("SHIB-USD") == "SHIB-USD"
    assert stocktwits._stocktwits_symbol("SHIB-USD", "crypto") == "SHIB.X"
    assert stocktwits._stocktwits_symbol("BTC-USD") == "BTC.X"


@pytest.mark.unit
@pytest.mark.parametrize("ticker, asset_type, expected", [
    ("AAPL", "stock", reddit.DEFAULT_SUBREDDITS),
    ("BTC-USD", "crypto", reddit.COIN_SUBREDDITS["BTC"]),
    ("SHIB-USD", "crypto", reddit.CRYPTO_SUBREDDITS),
    ("ETH-USD", None, reddit.COIN_SUBREDDITS["ETH"]),
])
def test_reddit_subreddits_follow_the_instrument(ticker, asset_type, expected):
    assert reddit.subreddits_for(ticker, asset_type) == expected


@pytest.mark.unit
def test_the_crypto_sentiment_prompt_names_crypto_communities(monkeypatch):
    from tradingagents.agents.analysts import sentiment_analyst

    crypto = sentiment_analyst._build_system_message(
        ticker="BTC-USD", start_date="2026-09-01", end_date="2026-09-08",
        news_block="n", stocktwits_block="s", reddit_block="r", asset_type="crypto",
    )
    stock = sentiment_analyst._build_system_message(
        ticker="NVDA", start_date="2026-09-01", end_date="2026-09-08",
        news_block="n", stocktwits_block="s", reddit_block="r",
    )
    assert "r/Bitcoin" in crypto and "r/wallstreetbets" not in crypto
    assert "token unlocks" in crypto and "earnings" not in crypto
    assert "r/wallstreetbets" in stock
