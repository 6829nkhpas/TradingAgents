"""The structured conviction tolerates sloppy model output instead of failing."""

import pytest

from tradingagents.agents.schemas import AnalystConviction, ConvictionDriver, conviction_label


@pytest.mark.unit
@pytest.mark.parametrize("raw, expected", [
    (72, 72), (72.6, 73), ("64", 64), (0, 1), (-5, 1), (150, 100), ("n/a", 50), (None, 50),
])
def test_score_is_clamped_to_1_100(raw, expected):
    assert AnalystConviction(score=raw).score == expected


@pytest.mark.unit
@pytest.mark.parametrize("raw, expected", [
    (0.7, 0.7), ("high", 0.85), ("Medium", 0.6), ("low", 0.35), (85, 0.85), ("70%", 0.7),
    (-1, 0.0), (None, 0.5), ("unsure", 0.5),
])
def test_confidence_accepts_words_and_percentages(raw, expected):
    assert AnalystConviction(score=50, confidence=raw).confidence == expected


@pytest.mark.unit
def test_long_text_is_truncated_not_rejected():
    conviction = AnalystConviction(
        score=55, thesis="x" * 400, risks=["r" * 500] * 8,
        drivers=[{"headline": "h" * 300, "impact": "bullish", "weight": 4}] * 9,
    )
    assert len(conviction.thesis) <= 150
    assert len(conviction.risks) == 5 and all(len(r) <= 200 for r in conviction.risks)
    assert len(conviction.drivers) == 5
    assert conviction.drivers[0].impact.value == "positive"
    assert conviction.drivers[0].weight == 0.04  # read as a percentage


@pytest.mark.unit
def test_a_driver_without_a_headline_is_dropped():
    conviction = AnalystConviction(score=50, drivers=[{"impact": "negative"}, {"headline": "kept"}])
    assert [d.headline for d in conviction.drivers] == ["kept"]
    assert conviction.drivers[0].impact.value == "neutral"


@pytest.mark.unit
@pytest.mark.parametrize("raw, expected", [
    ("short-term", "short-term"), ("Short term", "short-term"), ("medium_term", "medium-term"),
    ("long-term", "medium-term"), ("intraday", "intraday"), ("weeks", "short-term"), ("??", "short-term"),
])
def test_horizon_is_normalised(raw, expected):
    assert AnalystConviction(score=50, horizon=raw).horizon.value == expected


@pytest.mark.unit
@pytest.mark.parametrize("score, label", [(100, "Bullish"), (60, "Bullish"), (59, "Neutral"),
                                          (41, "Neutral"), (40, "Bearish"), (1, "Bearish")])
def test_label_thresholds(score, label):
    assert conviction_label(score) == label
    assert AnalystConviction(score=score).label == label


@pytest.mark.unit
def test_payload_has_every_field_and_no_nulls():
    payload = AnalystConviction(score=66, confidence=0.6, thesis="t",
                                drivers=[ConvictionDriver(headline="d", impact="positive")]).to_payload()
    assert payload == {
        "score": 66, "label": "Bullish", "confidence": 0.6, "thesis": "t",
        "drivers": [{"headline": "d", "impact": "positive", "weight": 0.5}],
        "risks": [], "horizon": "short-term",
    }


@pytest.mark.unit
def test_scorer_fills_an_empty_thesis_and_handles_failure():
    from unittest.mock import MagicMock

    from tradingagents.agents.scorer import create_conviction_scorer

    structured = MagicMock()
    structured.invoke.return_value = AnalystConviction(score=30, drivers=[{"headline": "Guidance cut"}])
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    score = create_conviction_scorer(llm, "News Analyst")

    conviction = score("a report", instrument_context="ctx", trade_date="2026-09-01")
    assert conviction.thesis == "Guidance cut"
    assert score("   ") is None                       # nothing to score

    structured.invoke.side_effect = ValueError("bad JSON")
    assert score("a report") is None                  # a failed call is a miss, not a crash


@pytest.mark.unit
def test_scorer_prompt_forbids_tools_and_technical_analysis():
    from unittest.mock import MagicMock

    from tradingagents.agents.scorer import create_conviction_scorer
    from tradingagents.agents.structured import NO_EXTERNAL_TOOLS

    seen = {}
    structured = MagicMock()
    structured.invoke.side_effect = lambda messages: seen.setdefault("m", messages) and None
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    create_conviction_scorer(llm, "Sentiment Analyst")("report text")

    system = seen["m"][0].content
    assert NO_EXTERNAL_TOOLS in system and "technical" in system
    assert "report text" in seen["m"][1].content
