"""Research service: sentiment, news and fundamentals convictions for one instrument.

Runs the three analysts in parallel, scores each report into a structured
conviction, and combines those into one overall conviction. It does not make or
execute trades, and it does no technical (price/indicator) analysis.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from datetime import UTC, date, datetime
from typing import Any

from tradingagents.agents.context import build_instrument_context, resolve_instrument_identity
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.evidence import collecting
from tradingagents.dataflows.router import VENDOR_METHODS, get_category_for_method
from tradingagents.dataflows.symbols import (
    crypto_base,
    looks_like_crypto_symbol,
    normalize_crypto_symbol,
    normalize_symbol,
    safe_ticker_component,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.graph.setup import CRYPTO_ANALYSTS, STOCK_ANALYSTS, GraphSetup
from tradingagents.llm_clients import build_llm_kwargs, create_llm_client
from tradingagents.overall import NoConvictionError, compute_overall
from tradingagents.research_store import ResearchStore

logger = logging.getLogger(__name__)

ANALYSTS_BY_ASSET_TYPE = {"stock": STOCK_ANALYSTS, "crypto": CRYPTO_ANALYSTS}
DISCLAIMER = "Research and analytics only; not financial or investment advice."
BASE_NOTICE = "Configured sources may be unavailable or rate-limited; see each report for data gaps."
CRYPTO_NOTICE = (
    "Crypto fundamentals cover supply, valuation and network activity from CoinGecko and "
    "Yahoo Finance; there are no company financials or on-chain analytics."
)
NO_OVERALL_NOTICE = "Overall conviction unavailable: no analyst report could be scored."

_VENDOR_LABELS = {
    "alpha_vantage": "Alpha Vantage",
    "fred": "FRED",
    "polymarket": "Polymarket",
    "sec_edgar": "SEC EDGAR",
    "yfinance": "Yahoo Finance",
}


class InvalidResearchRequest(ValueError):
    """The requested instrument or as-of date is not valid for analysis."""


class ResearchUnavailable(RuntimeError):
    """No analyst produced a report; the message is safe to show to callers."""


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
    """Describe the configured platforms behind each analyst's report."""
    sentiment = _configured_vendors(config, ("get_news",)) + ["StockTwits", "Reddit"]
    if os.getenv("TYPESAFE_API_KEY"):
        sentiment.append("TypeSafe Jev (optional post screening)")

    news = _configured_vendors(
        config,
        ("get_news", "get_global_news", "get_macro_indicators", "get_prediction_markets"),
    )
    if asset_type == "crypto":
        fundamentals = ["CoinGecko", "Yahoo Finance"]
    else:
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
        "sentiment": list(dict.fromkeys(sentiment)),
        "news": news,
        "fundamentals": fundamentals,
    }


def _clean_report(value: Any) -> str:
    """A report as stripped text; empty when the analyst produced none."""
    if isinstance(value, list):
        value = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in value
        )
    return value.strip() if isinstance(value, str) else ""


def _analyst_result(
    *,
    report: str,
    conviction: dict,
    error: str | None,
    sources: list[str],
    evidence_items: list[dict],
    label: str,
) -> dict[str, Any]:
    """One analyst's API section. Fields that do not apply are left out, never null."""
    if not report:
        result: dict[str, Any] = {
            "status": "unavailable",
            "reason": error or f"The {label} produced no report.",
        }
    elif not conviction:
        result = {
            "status": "degraded",
            "reason": "The report could not be scored into a conviction.",
            "report": report,
        }
    else:
        result = {"status": "ok", "conviction": conviction, "report": report}
    if evidence_items and report:
        result["evidence"] = evidence_items
    if sources:
        result["sources"] = sources
    return result


class ResearchAnalyzer:
    """Run the sentiment, news and fundamentals analysts and combine their convictions."""

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
        # A tool analyst takes two steps a round, plus its report, wrap-up and score.
        if 2 * max_tool_rounds + 4 >= max_recur_limit:
            raise ValueError(
                f"max_tool_rounds={max_tool_rounds} needs max_recur_limit "
                f"above {2 * max_tool_rounds + 4}"
            )

        graph_setup = GraphSetup(self.llm, max_tool_rounds=max_tool_rounds)
        self.graphs = {
            asset_type: graph_setup.setup_research_graph(analysts).compile()
            for asset_type, analysts in ANALYSTS_BY_ASSET_TYPE.items()
        }
        self.store = ResearchStore(
            max_concurrent=int(self.config.get("max_concurrent_runs") or 1),
            path=self._store_path(),
            min_refresh_age=float(self.config.get("research_refresh_min_age_seconds") or 0),
        )

    def _store_path(self) -> str | None:
        if not self.config.get("research_persist_results"):
            return None
        return self.config.get("research_store_path") or os.path.join(
            self.config["data_cache_dir"], "research_results.json"
        )

    def analyze(
        self,
        symbol: str,
        *,
        asset_type: str | None = None,
        analysis_date: date | str | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-friendly research result for one ticker or crypto asset.

        Each symbol keeps its latest result. A request for a symbol that has one
        is answered from it at once (``cached: true``) and queues a fresh run in
        the background (``refreshing: true``), which replaces the stored result
        when it finishes. Only a symbol never researched before waits for a run.

        A request for today (the default) shares the symbol's latest result
        whatever day it was computed; its ``analysis_date`` says which. A past
        ``analysis_date`` is its own entry, computed once and not refreshed.
        """
        ticker, resolved_asset_type = normalize_research_instrument(symbol, asset_type)
        as_of_date = _canonical_analysis_date(analysis_date)
        latest = as_of_date == get_current_date()
        if latest:
            key = (ticker, resolved_asset_type, "latest")

            def compute() -> dict[str, Any]:
                # Dated when the run starts, so a refresh tomorrow researches tomorrow.
                return self._run(ticker, resolved_asset_type, _canonical_analysis_date(None))
        else:
            key = (ticker, resolved_asset_type, as_of_date)

            def compute() -> dict[str, Any]:
                return self._run(ticker, resolved_asset_type, as_of_date)

        hit = self.store.get(key, compute, refresh=latest)
        result = hit.value
        result["cached"] = hit.cached
        result["refreshing"] = hit.refreshing
        return result

    def _run_config(self, asset_type: str) -> dict[str, Any]:
        config = deepcopy(self.config)
        if asset_type == "crypto" and config.get("crypto_global_news_queries"):
            config["global_news_queries"] = list(config["crypto_global_news_queries"])
        return config

    def _run(self, ticker: str, asset_type: str, as_of_date: str) -> dict[str, Any]:
        identity = resolve_instrument_identity(ticker) if asset_type == "stock" else {}
        instrument_context = build_instrument_context(ticker, asset_type, identity, as_of_date)
        specs = build_analyst_execution_plan(ANALYSTS_BY_ASSET_TYPE[asset_type]).specs
        initial_state: dict[str, Any] = {
            "messages": [("human", f"Research {ticker} as of {as_of_date}")],
            "company_of_interest": ticker,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": as_of_date,
            "analyst_errors": {},
        }
        for spec in specs:
            initial_state[spec.report_key] = ""
            initial_state[spec.conviction_key] = {}

        run_cfg = self._run_config(asset_type)
        # Data tools read their vendors/configuration from a context-local copy,
        # so simultaneous API requests do not overwrite one another's settings;
        # the evidence collector is context-local the same way.
        with run_config(run_cfg), collecting() as collector:
            state = self.graphs[asset_type].invoke(
                initial_state,
                config={"recursion_limit": int(self.config["max_recur_limit"])},
            )

        errors = state.get("analyst_errors") or {}
        sources = configured_research_sources(run_cfg, asset_type=asset_type)
        analysts: dict[str, Any] = {}
        convictions: dict[str, dict] = {}
        for spec in specs:
            report = _clean_report(state.get(spec.report_key))
            conviction = state.get(spec.conviction_key) or {}
            analysts[spec.name] = _analyst_result(
                report=report,
                conviction=conviction,
                error=errors.get(spec.name),
                sources=sources.get(spec.name, []),
                evidence_items=collector.items_for(spec.name),
                label=spec.agent_node,
            )
            if report and conviction:
                convictions[spec.name] = conviction

        if all(a["status"] == "unavailable" for a in analysts.values()):
            reasons = "; ".join(a["reason"] for a in analysts.values())
            raise ResearchUnavailable(f"No analyst produced a report for {ticker}: {reasons}")

        notices = [BASE_NOTICE]
        if asset_type == "crypto":
            notices.append(CRYPTO_NOTICE)
        notices.extend(collector.notices)

        result: dict[str, Any] = {
            "symbol": ticker,
            "asset_type": asset_type,
            "analysis_date": as_of_date,
            # A string, so the stored result is plain JSON and survives a restart unchanged.
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "cached": False,
            "refreshing": False,
            "analysts": analysts,
            "notices": notices,
            "disclaimer": DISCLAIMER,
        }
        if asset_type == "crypto":
            base = crypto_base(ticker, any_base=True)
            if base:
                result["base_asset"] = base
        try:
            result["overall"] = compute_overall(convictions, self.config.get("conviction_weights"))
        except NoConvictionError:
            notices.append(NO_OVERALL_NOTICE)
        return result
