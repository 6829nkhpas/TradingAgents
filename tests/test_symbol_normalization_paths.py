"""Symbol normalization must apply on every yfinance path.

Regression tests for #983 (instrument identity) and the news path: a broker
symbol like XAUUSD must resolve to the canonical Yahoo symbol (GC=F), so
identity and news lookups hit the right instrument instead of failing/mismatching.
"""
import tradingagents.agents.context as au
import tradingagents.dataflows.vendors.yahoo.fundamentals as yahoo_fundamentals
import tradingagents.dataflows.vendors.yahoo.news as ynews


def test_identity_lookup_normalizes_symbol(monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        @property
        def info(self):
            return {"longName": "Gold Futures", "quoteType": "FUTURE"}

    monkeypatch.setattr(yahoo_fundamentals.yf, "Ticker", FakeTicker)
    au._identity.cache_clear()

    identity = au.resolve_instrument_identity("XAUUSD")

    assert seen["symbol"] == "GC=F"  # normalized, not the raw broker symbol
    assert identity.get("company_name") == "Gold Futures"


def test_news_lookup_normalizes_symbol(monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        def get_news(self, count):
            return []

    class EmptySearch:
        # An empty ticker feed falls back to a keyword search; it finds nothing too.
        def __init__(self, *a, **k):
            self.news = []

    monkeypatch.setattr(ynews.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(ynews.yf, "Search", EmptySearch)
    monkeypatch.setattr(ynews, "yf_retry", lambda fn: fn())

    out = ynews.get_news_yfinance("XAUUSD", "2025-01-01", "2025-01-10")

    assert seen["symbol"] == "GC=F"   # news queried with the canonical symbol
    assert "XAUUSD" in out            # the user's ticker stays in the report
    assert "GC=F" in out              # provenance noted
