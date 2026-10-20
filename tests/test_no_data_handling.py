"""Tests that empty vendor results never become fabricated data.

Covers:
  - route_to_vendor must convert NoMarketDataError into a single explicit
    "NO_DATA_AVAILABLE" sentinel after all vendors are exhausted.
"""

import unittest
from unittest import mock

import pandas as pd
import pytest

from tradingagents.dataflows import router
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.errors import NoMarketDataError


@pytest.mark.unit
class TestRouteToVendorSentinel(unittest.TestCase):
    def test_no_data_from_all_vendors_returns_sentinel(self):
        def raises_no_data(symbol, *a, **k):
            raise NoMarketDataError(symbol, "GC=F", "no rows")

        patched = {"yfinance": raises_no_data, "alpha_vantage": raises_no_data}
        with mock.patch.dict(
            router.VENDOR_METHODS, {"get_news": patched}, clear=False
        ):
            result = router.route_to_vendor(
                "get_news", "XAUUSD+", "2026-01-01", "2026-01-10"
            )
        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertIn("XAUUSD+", result)
        self.assertIn("GC=F", result)
        self.assertIn("Do not estimate", result)

    def test_unconfigured_fallback_does_not_mask_no_data(self):
        # When the primary vendor reports no data and the fallback is simply
        # unavailable (e.g. missing API key -> raises), the no-data sentinel
        # must win rather than the fallback's incidental error crashing out.
        def raises_no_data(symbol, *a, **k):
            raise NoMarketDataError(symbol, symbol, "no rows")

        def raises_unavailable(symbol, *a, **k):
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is not set.")

        patched = {"yfinance": raises_no_data, "alpha_vantage": raises_unavailable}
        with mock.patch.dict(
            router.VENDOR_METHODS, {"get_news": patched}, clear=False
        ):
            result = router.route_to_vendor(
                "get_news", "FAKE", "2026-01-01", "2026-01-10"
            )
        self.assertIn("NO_DATA_AVAILABLE", result)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.unit
def test_an_unreachable_yahoo_is_not_reported_as_a_symbol_without_insider_data():
    from tradingagents.dataflows.errors import VendorUnavailableError
    from tradingagents.dataflows.vendors.yahoo import fundamentals

    ticker = type("T", (), {"insider_transactions": pd.DataFrame()})()
    with mock.patch.object(fundamentals.yf, "Ticker", return_value=ticker), \
         mock.patch.object(fundamentals, "vendor_reachable", return_value=False), \
         pytest.raises(VendorUnavailableError):
        fundamentals.get_insider_transactions("AAPL", as_of_date=get_current_date())
