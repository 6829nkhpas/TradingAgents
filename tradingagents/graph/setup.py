import logging
from collections import Counter
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_conviction_scorer,
    create_crypto_fundamentals_analyst,
    create_fundamentals_analyst,
    create_news_analyst,
    create_sentiment_analyst,
)
from tradingagents.agents.analysts.turn import WRAP_UP, content_text
from tradingagents.agents.state import ResearchState
from tradingagents.dataflows.evidence import analyst_scope

from .analyst_execution import build_analyst_execution_plan

logger = logging.getLogger(__name__)

STOCK_ANALYSTS = ("social", "news", "fundamentals")
CRYPTO_ANALYSTS = ("social", "news", "crypto_fundamentals")


def _analyst_graph(spec, agent, scorer, max_tool_rounds: int, state_schema=ResearchState):
    """One analyst as a graph of its own: the model, its tools and its scorer, on a
    private message history.

    It returns only its report and the conviction scored from it, so analysts
    running side by side never write the same key, and its tool calls never reach
    the other analysts' messages. After ``max_tool_rounds`` rounds of tool calls
    it is told to write its report, and that turn ends its research whatever it
    answers, so a model that keeps calling tools cannot run the graph into its
    recursion limit (#1420).
    """
    output = TypedDict(
        f"{spec.key.title().replace('_', '')}Output",
        {spec.report_key: str, spec.conviction_key: dict},
    )
    graph = StateGraph(state_schema, output_schema=output)

    def score(state):
        report = content_text(state.get(spec.report_key)).strip()
        if not report:
            return {spec.report_key: "", spec.conviction_key: {}}
        conviction = scorer(
            report,
            instrument_context=state.get("instrument_context") or "",
            trade_date=state.get("trade_date") or "",
        )
        return {
            spec.report_key: report,
            spec.conviction_key: conviction.to_payload() if conviction is not None else {},
        }

    graph.add_node("agent", agent)
    graph.add_node("score", score)
    graph.add_edge(START, "agent")
    graph.add_edge("score", END)
    if not spec.tools:
        graph.add_edge("agent", "score")
        return graph.compile()

    def calls(messages):
        return [call["name"] for m in messages for call in (getattr(m, "tool_calls", None) or [])]

    def rounds(messages) -> int:
        return sum(1 for m in messages if getattr(m, "tool_calls", None))

    def tools_or_score(state) -> str:
        return "tools" if state["messages"][-1].tool_calls else "score"

    def more_or_wrap_up(state) -> str:
        return "wrap_up" if rounds(state["messages"]) >= max_tool_rounds else "agent"

    def wrap_up(state):
        repeated = ", ".join(f"{name} x{n}" for name, n in Counter(calls(state["messages"])).most_common())
        logger.warning("%s used its %d tool rounds (%s); asking for its report",
                       spec.agent_node, max_tool_rounds, repeated)
        return agent({**state, "messages": [*state["messages"], HumanMessage(WRAP_UP)]})

    graph.add_node("tools", ToolNode(list(spec.tools)))
    graph.add_node("wrap_up", wrap_up)
    graph.add_conditional_edges("agent", tools_or_score, ["tools", "score"])
    graph.add_conditional_edges("tools", more_or_wrap_up, ["agent", "wrap_up"])
    graph.add_edge("wrap_up", "score")
    return graph.compile()


def _isolated(spec, subgraph):
    """Run one analyst's subgraph so that its failure stays its own.

    Everything the analyst's data calls record as evidence is tagged with its
    section name. An exception (a vendor outage, a provider error, the recursion
    limit) is reported under ``analyst_errors`` instead of ending the whole
    research run, so the other analysts' reports still reach the caller.
    """

    def run(state, config):
        with analyst_scope(spec.name):
            try:
                out = subgraph.invoke(state, config)
            except Exception as exc:  # noqa: BLE001 — one analyst must not sink the run
                logger.exception("%s failed", spec.agent_node)
                return {"analyst_errors": {
                    spec.name: f"The {spec.agent_node} failed ({type(exc).__name__}).",
                }}
        return {
            spec.report_key: out.get(spec.report_key) or "",
            spec.conviction_key: out.get(spec.conviction_key) or {},
        }

    return run


class GraphSetup:
    """Builds the research graph: the selected analysts, run side by side."""

    def __init__(self, llm: Any, max_tool_rounds: int):
        self.llm = llm
        self.max_tool_rounds = max_tool_rounds

    def _build_analyst_nodes(self, selected_analysts):
        """One isolated, compiled analyst subgraph per selection, keyed by node name."""
        plan = build_analyst_execution_plan(selected_analysts)
        analyst_factories = {
            "social": lambda: create_sentiment_analyst(self.llm),
            "news": lambda: create_news_analyst(self.llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.llm),
            "crypto_fundamentals": lambda: create_crypto_fundamentals_analyst(self.llm),
        }
        nodes = {
            spec.agent_node: _isolated(spec, _analyst_graph(
                spec,
                analyst_factories[spec.key](),
                create_conviction_scorer(self.llm, spec.agent_node),
                self.max_tool_rounds,
            ))
            for spec in plan.specs
        }
        return plan, nodes

    def setup_research_graph(self, selected_analysts=STOCK_ANALYSTS):
        """Build a graph that runs the analysts in parallel and then ends."""
        _, analyst_nodes = self._build_analyst_nodes(selected_analysts)
        workflow = StateGraph(ResearchState)
        for name, node in analyst_nodes.items():
            workflow.add_node(name, node)

        analysts = list(analyst_nodes)
        for name in analysts:
            workflow.add_edge(START, name)
        workflow.add_edge(analysts, END)
        return workflow
