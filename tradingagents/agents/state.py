from typing import Annotated

from langgraph.graph import MessagesState


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer for a dict that parallel analysts each add their own key to."""
    return {**(left or {}), **(right or {})}


class AnalystState(MessagesState):
    """The minimum shared context passed into each individual analyst graph."""

    company_of_interest: Annotated[str, "Ticker or crypto symbol under analysis"]
    asset_type: Annotated[str, "Asset type under analysis: stock or crypto"]
    instrument_context: Annotated[str, "Resolved ticker identity and asset context"]
    trade_date: Annotated[str, "The analysis date; data is served as of it"]


class ResearchState(AnalystState):
    """State used by the research graph: one report and conviction per analyst."""

    sentiment_report: Annotated[str, "Report from the Sentiment Analyst"]
    news_report: Annotated[str, "Report from the News Analyst"]
    fundamentals_report: Annotated[str, "Report from the Fundamentals Analyst"]
    sentiment_conviction: Annotated[dict, "Conviction scored from the sentiment report"]
    news_conviction: Annotated[dict, "Conviction scored from the news report"]
    fundamentals_conviction: Annotated[dict, "Conviction scored from the fundamentals report"]
    # Why an analyst produced nothing, keyed by its section name. Written by
    # analysts running in parallel, so it merges instead of overwriting.
    analyst_errors: Annotated[dict, merge_dicts]
