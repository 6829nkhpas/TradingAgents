"""Source evidence: recorded by the vendors, tagged per analyst, deduplicated and capped."""

from __future__ import annotations

import json
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from tradingagents.dataflows import evidence


@pytest.mark.unit
def test_recording_outside_a_run_is_a_no_op():
    evidence.record(type="article", title="t", source="s")   # must not raise
    with evidence.collecting() as collector:
        evidence.record(type="article", title="t", source="s")  # no analyst in scope
    assert collector.items_for("news") == []


@pytest.mark.unit
def test_items_are_deduplicated_newest_first_capped_and_numbered():
    with evidence.collecting() as collector, evidence.analyst_scope("news"):
        evidence.record(type="article", title="old", source="A", url="https://a/1",
                        published_at="2026-09-01T00:00:00Z")
        evidence.record(type="article", title="new", source="B", url="https://a/2",
                        published_at="2026-09-03T00:00:00Z")
        evidence.record(type="article", title="old again", source="A", url="https://a/1")
        evidence.record(type="article", title="undated", source="C")
        evidence.record(type="article", title="UNDATED", source="c")       # same title+source
        for i in range(30):
            evidence.record(type="article", title=f"filler {i}", source="D", url=f"https://d/{i}")

    items = collector.items_for("news")
    assert len(items) == evidence.DEFAULT_CAP
    assert [i["title"] for i in items[:2]] == ["new", "old"]
    assert items[2]["title"] == "undated"
    assert [i["id"] for i in items[:3]] == ["e1", "e2", "e3"]


@pytest.mark.unit
def test_an_item_needs_a_title_and_source_and_only_keeps_real_links():
    with evidence.collecting() as collector, evidence.analyst_scope("sentiment"):
        evidence.record(type="social_post", title="", source="Reddit")
        evidence.record(type="social_post", title="post", source="")
        evidence.record(type="social_post", title="post", source="Reddit", url="javascript:alert(1)",
                        published_at="not a date", stance="")
    [item] = collector.items_for("sentiment")
    assert item == {"id": "e1", "type": "social_post", "title": "post", "source": "Reddit"}


@pytest.mark.unit
def test_an_unknown_type_is_rejected():
    with evidence.collecting(), evidence.analyst_scope("news"), pytest.raises(ValueError):
        evidence.record(type="opinion", title="t", source="s")


@pytest.mark.unit
def test_publish_times_are_normalised_to_utc_iso():
    from datetime import UTC, datetime

    assert evidence._iso(datetime(2026, 9, 1, 12, tzinfo=UTC)) == "2026-09-01T12:00:00Z"
    assert evidence._iso(1788264000) == "2026-09-01T12:00:00Z"
    assert evidence._iso("2026-09-01T12:00:00+00:00") == "2026-09-01T12:00:00Z"
    assert evidence._iso("yesterday") is None


@pytest.mark.unit
def test_tool_calls_inside_parallel_analysts_are_tagged_to_their_analyst(monkeypatch):
    """The news analyst's evidence is recorded inside a ToolNode thread; the
    sentiment analyst's inside its own node. Each must land under its own name."""
    from tradingagents.agents import tools
    from tradingagents.agents.analysts import (
        crypto_fundamentals_analyst as crypto_fund,
        sentiment_analyst as sentiment,
    )
    from tradingagents.agents.schemas import AnalystConviction, SentimentBand, SentimentReport
    from tradingagents.graph.setup import CRYPTO_ANALYSTS, GraphSetup

    def routed(method, *args, **kwargs):
        evidence.record(type="article", title=f"{method} article", source="Yahoo",
                        url=f"https://news/{method}")
        return f"{method} data"

    # get_news serves both the sentiment analyst (called directly) and the news
    # analyst (called as a tool), so the same article lands under each of them.
    monkeypatch.setattr(tools, "route_to_vendor", routed)
    monkeypatch.setattr(sentiment, "fetch_reddit_posts", lambda *a, **k: "rd")

    def stocktwits(*a, **k):
        evidence.record(type="social_post", title="post", source="StockTwits", url="https://st/1")
        return "st"

    monkeypatch.setattr(sentiment, "fetch_stocktwits_messages", stocktwits)

    def coingecko(*a, **k):
        evidence.record(type="data", title="BTC on CoinGecko", source="CoinGecko", url="https://cg/btc")
        return "cg"

    monkeypatch.setattr(crypto_fund, "fetch_coingecko_fundamentals", coingecko)
    monkeypatch.setattr(crypto_fund, "_yahoo_block", lambda *a: "yf")

    class LLM:
        def bind_tools(self, tools_):
            def reply(prompt):
                messages = prompt.to_messages()
                if any(isinstance(m, ToolMessage) for m in messages):
                    return AIMessage(content="news report")
                return AIMessage(content="", tool_calls=[{
                    "name": "get_news", "id": "c1",
                    "args": {"start_date": "2026-09-01", "end_date": "2026-09-08"},
                }])
            return RunnableLambda(reply)

        def invoke(self, messages):
            return AIMessage(content="crypto report")

        def with_structured_output(self, schema):
            if schema is SentimentReport:
                return RunnableLambda(lambda _: SentimentReport(
                    overall_band=SentimentBand.NEUTRAL, overall_score=5, confidence="low", narrative="n"))
            return RunnableLambda(lambda _: AnalystConviction(score=55))

    graph = GraphSetup(LLM(), max_tool_rounds=3).setup_research_graph(CRYPTO_ANALYSTS).compile()
    state = {"messages": [("human", "go")], "company_of_interest": "BTC-USD", "asset_type": "crypto",
             "instrument_context": "", "trade_date": "2026-09-08", "analyst_errors": {}}
    with evidence.collecting() as collector:
        out = graph.invoke(state)

    assert out["news_report"] == "news report" and out["news_conviction"]["score"] == 55
    assert [i["title"] for i in collector.items_for("news")] == ["get_news article"]
    assert [i["title"] for i in collector.items_for("sentiment")] == ["get_news article", "post"]
    assert [i["source"] for i in collector.items_for("fundamentals")] == ["CoinGecko"]


@pytest.mark.unit
def test_vendor_return_strings_are_unchanged_by_recording():
    from tradingagents.dataflows.vendors import stocktwits

    payload = {"messages": [{
        "id": 7, "created_at": "2026-09-08T10:00:00Z", "body": "to the moon",
        "user": {"username": "alice"}, "entities": {"sentiment": {"basic": "Bullish"}},
    }]}
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()

    with mock.patch.object(stocktwits, "urlopen", return_value=response):
        plain = stocktwits.fetch_stocktwits_messages("NVDA")
        with evidence.collecting() as collector, evidence.analyst_scope("sentiment"):
            recorded = stocktwits.fetch_stocktwits_messages("NVDA")

    assert plain == recorded
    [item] = collector.items_for("sentiment")
    assert item["url"] == "https://stocktwits.com/alice/message/7"
    assert item["stance"] == "Bullish" and item["published_at"] == "2026-09-08T10:00:00Z"


@pytest.mark.unit
def test_reddit_posts_carry_their_link(monkeypatch):
    from tradingagents.dataflows.vendors import reddit

    feed = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>BTC breaks out</title><published>2026-09-08T09:00:00+00:00</published>
      <category term="Bitcoin"/><link href="https://www.reddit.com/r/Bitcoin/comments/abc/x/"/>
      <content>body</content></entry></feed>"""
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = feed

    with mock.patch.object(reddit, "urlopen", return_value=response), \
            evidence.collecting() as collector, evidence.analyst_scope("sentiment"):
        out = reddit.fetch_reddit_posts("BTC-USD", asset_type="crypto")

    assert "BTC breaks out" in out
    [item] = collector.items_for("sentiment")
    assert item["url"] == "https://www.reddit.com/r/Bitcoin/comments/abc/x/"
    assert item["source"] == "Reddit r/Bitcoin"


@pytest.mark.unit
def test_polymarket_markets_link_to_their_event(monkeypatch):
    from tradingagents.dataflows.date_window import get_current_date
    from tradingagents.dataflows.vendors import polymarket

    data = {"events": [{"slug": "fed-decision-december", "markets": [{
        "question": "Fed cuts in December?", "outcomes": '["Yes","No"]', "outcomePrices": '["0.62","0.38"]',
        "volumeNum": 1000, "endDate": "2099-12-31T00:00:00Z", "closed": False,
    }]}]}
    monkeypatch.setattr(polymarket, "_request", lambda *a, **k: data)

    with evidence.collecting() as collector, evidence.analyst_scope("news"):
        polymarket.get_prediction_markets("Fed", as_of_date=get_current_date())

    [item] = collector.items_for("news")
    assert item["url"] == "https://polymarket.com/event/fed-decision-december"
    assert item["type"] == "prediction_market" and "62%" in item["title"]
