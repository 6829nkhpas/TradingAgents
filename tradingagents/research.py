"""Research-only analysis service backed by the project's analyst agents.

This service deliberately does not create the bull/bear debate, trader, risk
team, portfolio manager, memory log, or trade-decision pipeline.
"""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import UTC, date, datetime
from typing import Any

from tradingagents.agents.context import build_instrument_context, resolve_instrument_identity
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.router import VENDOR_METHODS, get_category_for_method
from tradingagents.dataflows.symbols import (
    looks_like_crypto_symbol,
    normalize_crypto_symbol,
    normalize_symbol,
    safe_ticker_component,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.setup import GraphSetup
from tradingagents.llm_clients import build_llm_kwargs, create_llm_client

DEFAULT_ANALYSTS = ("market", "social", "news", "fundamentals")
CRYPTO_ANALYSTS = ("market", "social", "news")
REPORT_KEYS = {
    "market": "market_report",
    "sentiment": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}

_VENDOR_LABELS = {
    "alpha_vantage": "Alpha Vantage",
    "fred": "FRED",
    "polymarket": "Polymarket",
    "sec_edgar": "SEC EDGAR",
    "yfinance": "Yahoo Finance",
}


class InvalidResearchRequest(ValueError):
    """The requested instrument or as-of date is not valid for analysis."""


def normalize_research_instrument(
    symbol: str, asset_type: str | None = None
) -> tuple[str, str]:
    """Resolve an API ticker to its canonical symbol and asset type.

    Crypto pairs and common coin names are inferred when possible. Set
    ``asset_type="crypto"`` for a less common coin or an ambiguous bare ticker.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise InvalidResearchRequest("symbol must be a non-empty ticker or crypto name")

    requested = symbol.strip()
    resolved_type = asset_type or ("crypto" if looks_like_crypto_symbol(requested) else "stock")
    if resolved_type not in {"stock", "crypto"}:
        raise InvalidResearchRequest("asset_type must be 'stock' or 'crypto'")

    try:
        canonical = (
            normalize_crypto_symbol(requested)
            if resolved_type == "crypto"
            else normalize_symbol(requested)
        )
        canonical = safe_ticker_component(canonical)
    except ValueError as exc:
        raise InvalidResearchRequest(str(exc)) from exc

    return canonical, resolved_type


def _canonical_analysis_date(value: date | str | None) -> str:
    """Return a canonical as-of date no later than today."""
    today = date.fromisoformat(get_current_date())
    if value is None:
        return today.isoformat()

    try:
        if isinstance(value, datetime):
            parsed = value.date()
        else:
            parsed = value if isinstance(value, date) else date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise InvalidResearchRequest("analysis_date must use YYYY-MM-DD format") from exc

    if parsed > today:
        raise InvalidResearchRequest(f"analysis_date cannot be in the future: {parsed.isoformat()}")
    return parsed.isoformat()


def _merge_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Copy defaults and apply a one-level merge for nested configuration maps."""
    config = deepcopy(DEFAULT_CONFIG)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(deepcopy(value))
        else:
            config[key] = deepcopy(value)
    return config


def _configured_vendors(config: dict[str, Any], methods: tuple[str, ...]) -> list[str]:
    """Vendor platforms configured for a set of tools, in routing order.

    ``default`` means the router may try every implementation it knows for that
    tool. These are configured/available sources, not a guarantee that every
    upstream request succeeded in a particular run.
    """
    labels: list[str] = []
    tool_vendors = config.get("tool_vendors") or {}
    data_vendors = config.get("data_vendors") or {}

    for method in methods:
        implementations = VENDOR_METHODS.get(method, {})
        category = get_category_for_method(method)
        configured = tool_vendors.get(method) or data_vendors.get(category, "default")
        vendor_names = [part.strip() for part in str(configured).split(",") if part.strip()]

        if not vendor_names or vendor_names == ["default"]:
            vendor_names = list(implementations)

        for vendor in vendor_names:
            if vendor in implementations:
                label = _VENDOR_LABELS.get(vendor, vendor)
                if label not in labels:
                    labels.append(label)

    return labels


def configured_research_sources(
    config: dict[str, Any], *, asset_type: str
) -> dict[str, list[str]]:
    """Describe the configured platforms behind each research report."""
    market = _configured_vendors(config, ("get_stock_data", "get_indicators"))
    # The market analyst also calls the deterministic snapshot, which currently
    # uses Yahoo Finance directly even when a different primary vendor is set.
    if "Yahoo Finance" not in market:
        market.append("Yahoo Finance")

    sentiment = _configured_vendors(config, ("get_news",)) + ["StockTwits", "Reddit"]
    if os.getenv("TYPESAFE_API_KEY"):
        sentiment.append("TypeSafe Jev (optional post screening)")

    news = _configured_vendors(
        config,
        ("get_news", "get_global_news", "get_macro_indicators", "get_prediction_markets"),
    )
    fundamentals = _configured_vendors(
        config,
        (
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement",
            "get_insider_transactions",
        ),
    )

    return {
        "market": market,
        "sentiment": list(dict.fromkeys(sentiment)),
        "news": news,
        "fundamentals": fundamentals if asset_type == "stock" else [],
    }


class ResearchAnalyzer:
    """Run market, news, sentiment and (for stocks) fundamentals analysis."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = _merge_config(config)
        llm_kwargs = build_llm_kwargs(self.config)
        client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        self.llm = client.get_llm()

        max_tool_rounds = int(self.config["max_tool_rounds"])
        max_recur_limit = int(self.config["max_recur_limit"])
        if 2 * max_tool_rounds + 2 >= max_recur_limit:
            raise ValueError(
                f"max_tool_rounds={max_tool_rounds} needs max_recur_limit "
                f"above {2 * max_tool_rounds + 2}"
            )

        graph_setup = GraphSetup(
            self.llm,
            self.llm,
            conditional_logic=None,
            max_tool_rounds=max_tool_rounds,
        )
        self.graphs = {
            "stock": graph_setup.setup_research_graph(DEFAULT_ANALYSTS).compile(),
            "crypto": graph_setup.setup_research_graph(CRYPTO_ANALYSTS).compile(),
        }

    def analyze(
        self,
        symbol: str,
        *,
        asset_type: str | None = None,
        analysis_date: date | str | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-friendly research result for one ticker or crypto asset."""
        ticker, resolved_asset_type = normalize_research_instrument(symbol, asset_type)
        as_of_date = _canonical_analysis_date(analysis_date)
        identity = resolve_instrument_identity(ticker) if resolved_asset_type == "stock" else {}
        instrument_context = build_instrument_context(
            ticker,
            resolved_asset_type,
            identity,
            as_of_date,
        )
        initial_state = {
            "messages": [("human", f"Research {ticker} as of {as_of_date}")],
            "company_of_interest": ticker,
            "asset_type": resolved_asset_type,
            "instrument_context": instrument_context,
            "trade_date": as_of_date,
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
        }

        # Data tools read their vendors/configuration from a context-local copy,
        # so simultaneous API requests do not overwrite one another's settings.
        with run_config(self.config):
            state = self.graphs[resolved_asset_type].invoke(
                initial_state,
                config={"recursion_limit": int(self.config["max_recur_limit"])},
            )

        reports = {
            report_name: _clean_report(state.get(state_key))
            for report_name, state_key in REPORT_KEYS.items()
        }
        notices = [
            "Configured sources may be unavailable or rate-limited; see each report for data gaps."
        ]
        if resolved_asset_type == "crypto":
            notices.append(
                "Issuer/company fundamentals are omitted for crypto; this version does not "
                "provide on-chain analytics."
            )

        return {
            "symbol": ticker,
            "asset_type": resolved_asset_type,
            "analysis_date": as_of_date,
            "generated_at": datetime.now(UTC),
            "research": reports,
            "configured_sources": configured_research_sources(
                self.config,
                asset_type=resolved_asset_type,
            ),
            "notices": notices,
            "disclaimer": "Research and analytics only; not financial or investment advice.",
        }


def _clean_report(value: Any) -> str | None:
    """Return a non-empty report string or None when that section was not run."""
    if not isinstance(value, str):
        return None
    report = value.strip()
    return report or None
