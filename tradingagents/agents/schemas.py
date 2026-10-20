"""Pydantic schemas used by agents that produce structured output.

Each analyst's natural-language report is what users read; structured output
is layered on where a consumer needs machine-readable fields:

- The Sentiment Analyst writes its report as a ``SentimentReport``.
- Every analyst's report is scored into an ``AnalystConviction`` (see
  ``tradingagents.agents.scorer``).

Each provider's native structured-output mode is used (json_schema for
OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic). Field
descriptions become the model's output instructions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(StrEnum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Downstream consumers read ``overall_band`` and ``overall_score`` without
    fragile regex fallbacks. ``narrative`` preserves the rich source-by-source
    analysis; ``render_sentiment_report`` prepends a deterministic header so the
    report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the reader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to markdown.

    The structured header (band + score + confidence) is prepended to the
    narrative so the report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])


# ---------------------------------------------------------------------------
# Analyst conviction (one per analyst report)
# ---------------------------------------------------------------------------
#
# Models do not always respect numeric ranges or length limits, and a hard
# pydantic constraint fails the whole object, so every field is normalised in a
# ``mode="before"`` validator instead: clamp, truncate, coerce. One sloppy field
# then costs nothing rather than discarding the score.

BULLISH_FROM = 60
BEARISH_TO = 40
THESIS_CHARS = 150
_TEXT_CHARS = 200
_MAX_ITEMS = 5
_WORD_CONFIDENCE = {"low": 0.35, "medium": 0.6, "moderate": 0.6, "high": 0.85}


def conviction_label(score: float) -> str:
    """Bullish at 60 and above, Bearish at 40 and below, Neutral between."""
    if score >= BULLISH_FROM:
        return "Bullish"
    if score <= BEARISH_TO:
        return "Bearish"
    return "Neutral"


def _truncate(value, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().rstrip("%").strip()
        try:
            return float(text)
        except ValueError:
            return None
    return None


class ConvictionImpact(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ConvictionHorizon(StrEnum):
    INTRADAY = "intraday"
    SHORT_TERM = "short-term"
    MEDIUM_TERM = "medium-term"


_IMPACT_WORDS = {
    "positive": "positive", "bullish": "positive", "up": "positive", "supportive": "positive",
    "negative": "negative", "bearish": "negative", "down": "negative", "adverse": "negative",
}
_HORIZON_WORDS = {
    "intraday": "intraday", "day": "intraday", "hours": "intraday",
    "short": "short-term", "shortterm": "short-term", "days": "short-term", "weeks": "short-term",
    "medium": "medium-term", "mediumterm": "medium-term", "months": "medium-term",
    "long": "medium-term", "longterm": "medium-term",
}


class ConvictionDriver(BaseModel):
    """One factor behind the conviction."""

    headline: str = Field(description="The factor in one short line, taken from the report.")
    impact: ConvictionImpact = Field(
        default=ConvictionImpact.NEUTRAL,
        description="Direction of the factor's effect: positive, negative or neutral.",
    )
    weight: float = Field(
        default=0.5,
        description="How much the factor drives the conviction, from 0 (minor) to 1 (decisive).",
    )

    @field_validator("headline", mode="before")
    @classmethod
    def _headline(cls, value):
        return _truncate(value, _TEXT_CHARS)

    @field_validator("impact", mode="before")
    @classmethod
    def _impact(cls, value):
        word = str(value or "").strip().lower()
        return _IMPACT_WORDS.get(word, "neutral")

    @field_validator("weight", mode="before")
    @classmethod
    def _weight(cls, value):
        number = _to_number(value)
        if number is None:
            return 0.5
        if number > 1:
            number /= 100
        return round(_clamp(number, 0.0, 1.0), 2)


class AnalystConviction(BaseModel):
    """A directional conviction distilled from one analyst's report."""

    score: int = Field(
        description=(
            "Conviction on a 1-100 scale: 50 is neutral, 100 maximally bullish, "
            "1 maximally bearish. Use roughly 60-100 for bullish, 41-59 for "
            "neutral or mixed, 1-40 for bearish."
        ),
    )
    confidence: float = Field(
        default=0.5,
        description=(
            "Confidence in the score from 0 to 1, reflecting how much substantive, "
            "recent evidence the report has. Low when data was missing or sparse."
        ),
    )
    thesis: str = Field(
        default="",
        description="The conviction in one sentence of at most 150 characters.",
    )
    drivers: list[ConvictionDriver] = Field(
        default_factory=list,
        description="Up to 5 factors from the report that drive the score, strongest first.",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Up to 5 risks or caveats from the report that could invalidate the view.",
    )
    horizon: ConvictionHorizon = Field(
        default=ConvictionHorizon.SHORT_TERM,
        description="Time frame the evidence speaks to: intraday, short-term or medium-term.",
    )

    @field_validator("score", mode="before")
    @classmethod
    def _score(cls, value):
        number = _to_number(value)
        if number is None:
            return 50
        return int(round(_clamp(number, 1, 100)))

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, value):
        if isinstance(value, str) and value.strip().lower() in _WORD_CONFIDENCE:
            return _WORD_CONFIDENCE[value.strip().lower()]
        number = _to_number(value)
        if number is None:
            return 0.5
        if number > 1:
            number /= 100
        return round(_clamp(number, 0.0, 1.0), 2)

    @field_validator("thesis", mode="before")
    @classmethod
    def _thesis(cls, value):
        return _truncate(value, THESIS_CHARS)

    @field_validator("drivers", mode="before")
    @classmethod
    def _drivers(cls, value):
        if not isinstance(value, list):
            return []
        kept = [
            d for d in value
            if isinstance(d, ConvictionDriver)
            or (isinstance(d, dict) and str(d.get("headline") or "").strip())
        ]
        return kept[:_MAX_ITEMS]

    @field_validator("risks", mode="before")
    @classmethod
    def _risks(cls, value):
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        cleaned = [_truncate(r, _TEXT_CHARS) for r in value if str(r or "").strip()]
        return cleaned[:_MAX_ITEMS]

    @field_validator("horizon", mode="before")
    @classmethod
    def _horizon(cls, value):
        word = str(value or "").strip().lower().replace(" ", "").replace("_", "")
        word = word.replace("-term", "term").replace("-", "")
        if word in ("intraday", "shortterm", "mediumterm"):
            return {"intraday": "intraday", "shortterm": "short-term",
                    "mediumterm": "medium-term"}[word]
        return _HORIZON_WORDS.get(word, "short-term")

    @property
    def label(self) -> str:
        return conviction_label(self.score)

    def to_payload(self) -> dict:
        """The API shape: every field plus the derived label."""
        return {
            "score": self.score,
            "label": self.label,
            "confidence": self.confidence,
            "thesis": self.thesis,
            "drivers": [d.model_dump(mode="json") for d in self.drivers],
            "risks": list(self.risks),
            "horizon": self.horizon.value,
        }
