"""A Yahoo rate limit is retried, then reported as a rate limit.

When the limit outlasts the retries, the agent must hear that the vendor is
throttled. "The symbol may be invalid or delisted" is a claim about the company
that nobody checked, and an exception out of a tool ends the run.
"""

import pytest
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from tradingagents.agents.tools import (
    get_global_news,
    get_insider_transactions,
    get_news,
)
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors.yahoo import common, fundamentals

DAY = "2026-09-18"


def _rate_limited(*args, **kwargs):
    raise YFRateLimitError()


@pytest.fixture
def yahoo(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(common.time, "sleep", lambda seconds: None)
    for module in (common, fundamentals):
        monkeypatch.setattr(module, "vendor_reachable", lambda url: True)
    return monkeypatch


@pytest.mark.unit
@pytest.mark.parametrize("tool, args", [
    pytest.param(get_news, ("AAPL", "2026-09-10", DAY), id="news"),
    pytest.param(get_global_news, (DAY, 7, 5), id="global_news"),
    pytest.param(get_insider_transactions, ("AAPL",), id="insider"),
])
def test_a_rate_limit_that_outlasts_the_retries_is_reported_as_one(yahoo, tool, args):
    yahoo.setattr(yf.Ticker, "get_news", _rate_limited)
    yahoo.setattr(yf.Ticker, "insider_transactions", property(_rate_limited))
    yahoo.setattr(yf, "Search", _rate_limited)

    # Dated today: a past run is not served insider trades at all.
    out = tool.func(*args, trade_date=get_current_date())

    assert out.startswith("DATA_UNAVAILABLE"), out
    assert "delisted" not in out


@pytest.mark.unit
def test_fundamentals_ask_a_new_ticker_after_a_rate_limit(yahoo):
    """A Ticker keeps a failed ``info`` fetch as done, so asking the same one
    again reads an empty profile, which looks like a symbol with no data."""

    class Ticker:
        def __init__(self, symbol):
            self.fetched = False

        @property
        def info(self):
            if self.fetched:
                return {}
            self.fetched = True
            raise YFRateLimitError()

    yahoo.setattr(yf, "Ticker", Ticker)

    out = router.route_to_vendor("get_fundamentals", "AAPL", None)

    assert out.startswith("DATA_UNAVAILABLE"), out


class _Unreachable:
    """A Ticker whose every request fails in transit."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        raise TimeoutError("read timed out")


def _yahoo_calls():
    from tradingagents.dataflows.date_window import get_current_date
    from tradingagents.dataflows.vendors.yahoo import news

    today = get_current_date()
    return [
        pytest.param(lambda: fundamentals.get_fundamentals("AAPL", today), id="fundamentals"),
        pytest.param(lambda: fundamentals.get_balance_sheet("AAPL", "quarterly", today), id="statement"),
        pytest.param(lambda: fundamentals.get_insider_transactions("AAPL", today), id="insider"),
        pytest.param(lambda: fundamentals.get_company_profile("AAPL"), id="profile"),
        pytest.param(lambda: fundamentals.get_crypto_profile("BTC-USD", today), id="crypto_profile"),
        pytest.param(lambda: news.get_news_yfinance("AAPL", "2026-09-10", DAY), id="news"),
        pytest.param(lambda: news.get_global_news_yfinance(DAY, 7, 5), id="global_news"),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("call", _yahoo_calls())
def test_a_yahoo_request_that_fails_is_unavailable_not_absent(yahoo, call):
    """An unknown symbol comes back empty, so a request that raised failed in transit."""
    yahoo.setattr(yf, "Ticker", _Unreachable)
    yahoo.setattr(yf, "Search", _Unreachable)

    with pytest.raises(VendorUnavailableError):
        call()


def _yahoo_request_raises(yahoo, exc):
    """Every request yfinance makes raises ``exc``, below its own error handling."""
    from yfinance.data import YfData

    def request(self, *args, **kwargs):
        raise exc

    yahoo.setattr(YfData, "_make_request", request)


@pytest.mark.unit
@pytest.mark.parametrize("call", _yahoo_calls())
def test_a_request_that_yfinance_would_swallow_is_unavailable(yahoo, call):
    """yfinance answers some failed requests with an empty result; an empty
    statement from a Yahoo whose host answers a probe read as "no data"."""
    _yahoo_request_raises(yahoo, TimeoutError("read timed out"))

    with pytest.raises(VendorUnavailableError):
        call()


@pytest.mark.unit
@pytest.mark.parametrize("call", [c for c in _yahoo_calls() if c.id in ("fundamentals", "insider")])
def test_yahoo_answering_not_found_is_no_data(yahoo, call):
    """Yahoo answers an unknown symbol with HTTP 404: an answer about the symbol, not an outage."""
    from curl_cffi.requests import Response
    from curl_cffi.requests.exceptions import HTTPError

    from tradingagents.dataflows.errors import NoMarketDataError

    not_found = Response()
    not_found.status_code = 404
    _yahoo_request_raises(yahoo, HTTPError("HTTP Error 404: ", response=not_found))

    try:
        out = call()
    except NoMarketDataError:
        return
    assert out.startswith("No insider transactions"), out   # an empty filing list is not an error
