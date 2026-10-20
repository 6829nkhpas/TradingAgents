"""The overall conviction is a deterministic, explainable blend of the analysts'."""

import pytest

from tradingagents.overall import NoConvictionError, agreement_word, compute_overall


def _c(score, confidence=0.6, horizon="short-term", label=None):
    from tradingagents.agents.schemas import conviction_label

    return {"score": score, "confidence": confidence, "horizon": horizon,
            "label": label or conviction_label(score)}


@pytest.mark.unit
def test_hand_computed_blend():
    overall = compute_overall({
        "sentiment": _c(66, 0.6), "news": _c(61, 0.6), "fundamentals": _c(44, 0.6),
    })
    # Equal confidence: weights are the base weights 0.30 / 0.35 / 0.35.
    # score = 0.30*66 + 0.35*61 + 0.35*44 = 56.55 -> 57
    assert overall["score"] == 57
    assert overall["label"] == "Neutral"
    assert overall["weights"] == {"sentiment": 0.3, "news": 0.35, "fundamentals": 0.35}
    # pstdev(66, 61, 44) = 9.42 -> agreement 1 - 9.42/25 = 0.62
    assert overall["agreement"] == 0.62
    assert overall["confidence"] == round(0.6 * 0.62, 2)
    assert overall["components"] == ["sentiment", "news", "fundamentals"]
    assert overall["thesis"].startswith("Sentiment Bullish 66, News Bullish 61, Fundamentals Neutral 44")
    assert "analysts mixed" in overall["thesis"]


@pytest.mark.unit
def test_confidence_scales_each_weight():
    overall = compute_overall({"sentiment": _c(90, 0.9), "news": _c(30, 0.1)})
    # raw weights 0.30*0.9=0.27 and 0.35*0.1=0.035 -> 0.885 / 0.115
    assert overall["weights"] == {"sentiment": 0.885, "news": 0.115}
    assert overall["score"] == round(0.885 * 90 + 0.115 * 30)


@pytest.mark.unit
def test_a_single_analyst_fully_agrees_with_itself():
    overall = compute_overall({"news": _c(72, 0.8, horizon="medium-term")})
    assert overall["score"] == 72 and overall["label"] == "Bullish"
    assert overall["agreement"] == 1.0 and overall["confidence"] == 0.8
    assert overall["horizon"] == "medium-term"
    assert overall["weights"] == {"news": 1.0}


@pytest.mark.unit
def test_divergent_analysts_lower_agreement_and_confidence():
    aligned = compute_overall({"sentiment": _c(70, 0.8), "news": _c(72, 0.8)})
    split = compute_overall({"sentiment": _c(90, 0.8), "news": _c(20, 0.8)})
    assert aligned["agreement"] > 0.9 and agreement_word(aligned["agreement"]) == "analysts agree"
    assert split["agreement"] == 0.0 and agreement_word(split["agreement"]) == "analysts split"
    assert split["confidence"] < aligned["confidence"]


@pytest.mark.unit
def test_zero_confidence_still_counts_a_little():
    overall = compute_overall({"sentiment": _c(80, 0.0), "news": _c(40, 0.0)})
    assert 40 < overall["score"] < 80


@pytest.mark.unit
def test_horizon_tie_falls_back_to_short_term():
    overall = compute_overall({"sentiment": _c(50, horizon="intraday"),
                               "news": _c(50, horizon="medium-term")})
    assert overall["horizon"] == "short-term"


@pytest.mark.unit
def test_custom_base_weights_apply():
    overall = compute_overall({"sentiment": _c(80), "news": _c(40)},
                              {"sentiment": 1.0, "news": 0.0})
    assert overall["score"] == 80


@pytest.mark.unit
@pytest.mark.parametrize("convictions", [{}, {"news": {}}, {"news": {"label": "Bullish"}}])
def test_nothing_to_combine_is_an_error(convictions):
    with pytest.raises(NoConvictionError):
        compute_overall(convictions)
