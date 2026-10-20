from collections.abc import Iterable
from dataclasses import dataclass

from tradingagents.agents.analysts import fundamentals_analyst, news_analyst


@dataclass(frozen=True)
class AnalystNodeSpec:
    key: str             # selection key, e.g. "social"
    name: str            # API name of the analyst's section, e.g. "sentiment"
    agent_node: str      # graph node name
    report_key: str      # state key of the analyst's markdown report
    conviction_key: str  # state key of the structured conviction scored from it
    tools: tuple = ()


@dataclass(frozen=True)
class AnalystExecutionPlan:
    specs: list[AnalystNodeSpec]


ANALYST_NODE_SPECS: dict[str, AnalystNodeSpec] = {
    "social": AnalystNodeSpec(
        # Saved configs select this analyst as "social". It fetches its
        # sources before calling the model, so it has no tools.
        key="social",
        name="sentiment",
        agent_node="Sentiment Analyst",
        report_key="sentiment_report",
        conviction_key="sentiment_conviction",
    ),
    "news": AnalystNodeSpec(
        key="news",
        name="news",
        agent_node="News Analyst",
        report_key="news_report",
        conviction_key="news_conviction",
        tools=news_analyst.TOOLS,
    ),
    "fundamentals": AnalystNodeSpec(
        key="fundamentals",
        name="fundamentals",
        agent_node="Fundamentals Analyst",
        report_key="fundamentals_report",
        conviction_key="fundamentals_conviction",
        tools=fundamentals_analyst.TOOLS,
    ),
    "crypto_fundamentals": AnalystNodeSpec(
        # A coin's fundamentals: pre-fetched CoinGecko and Yahoo profiles, no tools.
        # It fills the same section as the equity fundamentals analyst.
        key="crypto_fundamentals",
        name="fundamentals",
        agent_node="Fundamentals Analyst",
        report_key="fundamentals_report",
        conviction_key="fundamentals_conviction",
    ),
}


def build_analyst_execution_plan(
    selected_analysts: Iterable[str],
) -> AnalystExecutionPlan:
    specs: list[AnalystNodeSpec] = []
    for analyst_key in selected_analysts:
        spec = ANALYST_NODE_SPECS.get(analyst_key)
        if spec is None:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        specs.append(spec)

    if not specs:
        raise ValueError("at least one analyst must be selected")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"two selected analysts fill the same section: {names}")

    return AnalystExecutionPlan(specs=specs)
