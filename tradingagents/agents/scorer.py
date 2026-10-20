"""Turn an analyst's finished report into a structured conviction.

Each analyst writes prose; the API's consumers need a number. A scorer makes one
structured-output call over the report alone (no tools, no new data), so the
score can only reflect what the analyst actually found.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.agents.schemas import AnalystConviction, conviction_label
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured

logger = logging.getLogger(__name__)

# The longest report the scorer reads; the conclusions sit at the end of a
# report (summary table), so a cut keeps the tail rather than the head.
_MAX_REPORT_CHARS = 24_000

Scorer = Callable[..., AnalystConviction | None]


def _scoring_prompt(analyst_label: str) -> str:
    return (
        f"You score a {analyst_label} report into a structured conviction about the "
        "instrument. Read the report and fill every field.\n"
        "- score: 1-100, where 50 is neutral, above 50 bullish and below 50 bearish. "
        "Score only what the report supports; when it says data was unavailable, "
        "move toward 50 and lower the confidence instead of guessing.\n"
        "- confidence: 0 to 1, from the amount, recency and agreement of the evidence.\n"
        "- thesis: one sentence of at most 150 characters.\n"
        "- drivers: up to 5 factors from the report, strongest first, each with its "
        "impact (positive / negative / neutral) and a weight from 0 to 1.\n"
        "- risks: up to 5 caveats from the report.\n"
        "- horizon: intraday, short-term or medium-term.\n"
        "Do not make price predictions or use technical/chart analysis. "
        + NO_EXTERNAL_TOOLS
    )


def create_conviction_scorer(llm, analyst_label: str) -> Scorer:
    """A scorer for one analyst; returns ``None`` when no conviction can be produced."""
    structured = bind_structured(llm, AnalystConviction, f"{analyst_label} scorer")
    system = _scoring_prompt(analyst_label)

    def score(report: str, *, instrument_context: str = "", trade_date: str = "") -> AnalystConviction | None:
        if structured is None or not report.strip():
            return None
        text = report if len(report) <= _MAX_REPORT_CHARS else report[-_MAX_REPORT_CHARS:]
        header = " ".join(part for part in (
            f"Analysis date: {trade_date}." if trade_date else "",
            instrument_context,
        ) if part)
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"{header}\n\n<report>\n{text}\n</report>".strip()),
        ]
        conviction = invoke_structured(structured, messages, f"{analyst_label} scorer")
        if conviction is None:
            return None
        if not isinstance(conviction, AnalystConviction):
            try:
                conviction = AnalystConviction.model_validate(conviction)
            except Exception as exc:  # noqa: BLE001 — a malformed answer is a miss, not a crash
                logger.warning("%s scorer returned an unusable object: %s", analyst_label, exc)
                return None
        if not conviction.thesis:
            lead = conviction.drivers[0].headline if conviction.drivers else ""
            conviction.thesis = (lead or f"{conviction_label(conviction.score)} reading of the "
                                 f"{analyst_label.lower()} report.")[:150]
        return conviction

    return score
