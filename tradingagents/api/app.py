"""FastAPI application exposing the research-only analyst workflow."""

from __future__ import annotations

import hmac
import logging
import os
from datetime import date, datetime
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import tradingagents
from tradingagents.research import InvalidResearchRequest, ResearchAnalyzer

logger = logging.getLogger(__name__)


class ResearchRequest(BaseModel):
    """One stock ticker or crypto symbol to analyze."""

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Ticker/symbol, e.g. AAPL, NVDA, BTC-USD, or Bitcoin.",
        examples=["AAPL", "BTC-USD"],
    )
    asset_type: Literal["stock", "crypto"] | None = Field(
        default=None,
        description="Optional for common crypto symbols; set to crypto for less common coins.",
    )
    analysis_date: date | None = Field(
        default=None,
        description="Optional as-of date in YYYY-MM-DD format; defaults to today.",
    )

    class Config:
        extra = "forbid"


class ResearchReports(BaseModel):
    market: str | None = None
    sentiment: str | None = None
    news: str | None = None
    fundamentals: str | None = None


class ResearchResponse(BaseModel):
    symbol: str
    asset_type: Literal["stock", "crypto"]
    analysis_date: date
    generated_at: datetime
    research: ResearchReports
    configured_sources: dict[str, list[str]]
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
        "Research-only stock and crypto analysis. The API returns market, "
        "sentiment, news, and (for stocks) fundamental analyst reports; it "
        "does not produce or execute trades."
    ),
)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Lightweight liveness check; it does not call an LLM or data provider."""
    return {"status": "ok", "service": "TradingAgents Research API"}


@app.post(
    "/api/v1/research",
    response_model=ResearchResponse,
    tags=["research"],
    summary="Analyze a stock or crypto symbol",
)
def research(
    request: ResearchRequest,
    _: None = Depends(require_api_key),
    analyzer: ResearchAnalyzer = Depends(get_research_analyzer),  # noqa: B008 — FastAPI DI default
) -> dict:
    """Run research analysts in parallel and return their full reports as JSON."""
    try:
        return analyzer.analyze(
            request.symbol,
            asset_type=request.asset_type,
            analysis_date=request.analysis_date,
        )
    except InvalidResearchRequest as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — upstream failures must not leak credentials
        logger.exception("Research request failed for symbol %r", request.symbol)
        raise HTTPException(
            status_code=502,
            detail=(
                "An upstream data or language-model provider failed; "
                "a complete research response could not be produced."
            ),
        ) from exc
