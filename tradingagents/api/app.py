"""FastAPI application exposing the research workflow.

``POST /api/v1/research`` returns one conviction per analyst (social sentiment,
news, fundamentals) plus an overall conviction, with each analyst's report and
the source evidence behind it. Fields that do not apply are omitted; the
response never contains nulls.
"""

from __future__ import annotations

import hmac
import logging
import os
from datetime import date, datetime
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import tradingagents
from tradingagents.research import InvalidResearchRequest, ResearchAnalyzer, ResearchUnavailable

logger = logging.getLogger(__name__)

Label = Literal["Bullish", "Bearish", "Neutral"]
Horizon = Literal["intraday", "short-term", "medium-term"]


class ResearchRequest(BaseModel):
    """One stock ticker or crypto symbol to analyze."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Ticker/symbol, e.g. AAPL, NVDA, BTC, BTC-USD, or Bitcoin.",
        examples=["AAPL", "BTC"],
    )
    asset_type: Literal["stock", "crypto"] | None = Field(
        default=None,
        description="Optional for common crypto symbols; set to crypto for less common coins.",
    )
    analysis_date: date | None = Field(
        default=None,
        description="Optional as-of date in YYYY-MM-DD format; defaults to today.",
    )


class Driver(BaseModel):
    headline: str
    impact: Literal["positive", "negative", "neutral"]
    weight: float = Field(ge=0, le=1)


class Conviction(BaseModel):
    score: int = Field(ge=1, le=100, description="1-100; 50 is neutral.")
    label: Label
    confidence: float = Field(ge=0, le=1)
    thesis: str
    drivers: list[Driver] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    horizon: Horizon


class EvidenceItem(BaseModel):
    id: str
    type: Literal["article", "social_post", "prediction_market", "data"]
    title: str
    source: str
    url: str | None = None
    published_at: str | None = None
    stance: str | None = None


class AnalystResult(BaseModel):
    status: Literal["ok", "degraded", "unavailable"] = Field(
        description=(
            "ok: report and conviction; degraded: report, but it could not be scored; "
            "unavailable: no report (see reason)."
        ),
    )
    conviction: Conviction | None = None
    report: str | None = Field(default=None, description="The analyst's markdown report.")
    reason: str | None = None
    evidence: list[EvidenceItem] | None = None
    sources: list[str] | None = None


class OverallConviction(BaseModel):
    score: int = Field(ge=1, le=100)
    label: Label
    confidence: float = Field(ge=0, le=1)
    agreement: float = Field(ge=0, le=1, description="1 when the analysts' scores agree.")
    thesis: str
    horizon: Horizon
    components: list[str]
    weights: dict[str, float]


class ResearchResponse(BaseModel):
    symbol: str
    base_asset: str | None = None
    asset_type: Literal["stock", "crypto"]
    analysis_date: date
    generated_at: datetime = Field(description="When this result was computed.")
    cached: bool = Field(description="Served from the symbol's stored result, not computed for this request.")
    refreshing: bool = Field(description="A newer run for the symbol is queued or running.")
    overall: OverallConviction | None = None
    analysts: dict[str, AnalystResult]
    notices: list[str]
    disclaimer: str


@lru_cache(maxsize=1)
def get_research_analyzer() -> ResearchAnalyzer:
    """Lazily create and reuse the compiled research graphs for this process."""
    try:
        return ResearchAnalyzer()
    except Exception as exc:  # noqa: BLE001 — report missing runtime configuration cleanly
        logger.exception("Could not initialize the research analyzer")
        raise HTTPException(
            status_code=503,
            detail=(
                "The research service is not configured. Set a valid LLM provider "
                "API key (for example OPENAI_API_KEY) and restart the server."
            ),
        ) from exc


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """Enforce a bearer key when TRADINGAGENTS_API_KEY is configured."""
    expected = os.getenv("TRADINGAGENTS_API_KEY", "").strip()
    if not expected:
        return

    scheme, separator, supplied = (authorization or "").partition(" ")
    if (
        scheme.lower() != "bearer"
        or not separator
        or not hmac.compare_digest(supplied.strip(), expected)
    ):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


app = FastAPI(
    title="TradingAgents Research API",
    version=tradingagents.__version__,
    description=(
        "Research-only stock and crypto analysis. Returns a social-sentiment, news and "
        "fundamentals conviction plus an overall conviction for a symbol; it does no "
        "technical analysis and does not produce or execute trades."
    ),
)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Lightweight liveness check; it does not call an LLM or data provider."""
    return {"status": "ok", "service": "TradingAgents Research API"}


@app.post(
    "/api/v1/research",
    response_model=ResearchResponse,
    response_model_exclude_none=True,
    tags=["research"],
    summary="Research convictions for a stock or crypto symbol",
)
def research(
    request: ResearchRequest,
    _: None = Depends(require_api_key),
    analyzer: ResearchAnalyzer = Depends(get_research_analyzer),  # noqa: B008 — FastAPI DI default
) -> dict:
    """Run the analysts in parallel and return their convictions, reports and evidence."""
    try:
        return analyzer.analyze(
            request.symbol,
            asset_type=request.asset_type,
            analysis_date=request.analysis_date,
        )
    except InvalidResearchRequest as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ResearchUnavailable as exc:
        logger.warning("Research unavailable for %r: %s", request.symbol, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — upstream failures must not leak credentials
        logger.exception("Research request failed for symbol %r", request.symbol)
        raise HTTPException(
            status_code=502,
            detail=(
                "An upstream data or language-model provider failed; "
                "a complete research response could not be produced."
            ),
        ) from exc
