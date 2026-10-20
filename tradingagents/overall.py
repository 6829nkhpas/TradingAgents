"""The overall (main) conviction, combined from the analysts' convictions.

Deterministic on purpose: no extra LLM call, the same inputs always give the
same answer, and every number can be traced back to the analysts.

- Each analyst's weight is its base weight (``conviction_weights`` in the config)
  times its own confidence (floored at 0.05), renormalised over the analysts
  that produced a conviction.
- ``score`` is the weighted mean of their scores (1-100, 50 neutral).
- ``agreement`` is ``1 - min(1, stdev(scores) / 25)``: 1 when every analyst gives
  the same score, 0 when they are a population standard deviation of 25 points
  or more apart (e.g. one at 25 and one at 75).
- ``confidence`` is the weighted mean confidence times the agreement, so analysts
  that contradict each other lower it.
- ``label`` uses the analysts' thresholds: Bullish at 60+, Bearish at 40 or below.
- ``horizon`` is the most common analyst horizon; a tie falls back to short-term.
"""

from __future__ import annotations

from collections import Counter
from statistics import pstdev

from tradingagents.agents.schemas import THESIS_CHARS, conviction_label

DEFAULT_WEIGHTS = {"sentiment": 0.30, "news": 0.35, "fundamentals": 0.35}
ANALYST_ORDER = ("sentiment", "news", "fundamentals")
_CONFIDENCE_FLOOR = 0.05
_AGREEMENT_SPREAD = 25.0


class NoConvictionError(ValueError):
    """No analyst produced a conviction, so there is nothing to combine."""


def agreement_word(agreement: float) -> str:
    if agreement >= 0.75:
        return "analysts agree"
    if agreement >= 0.45:
        return "analysts mixed"
    return "analysts split"


def _thesis(convictions: dict[str, dict], agreement: float) -> str:
    parts = [
        f"{name.capitalize()} {c['label']} {c['score']}"
        for name, c in convictions.items()
    ]
    text = f"{', '.join(parts)}; {agreement_word(agreement)}."
    return text if len(text) <= THESIS_CHARS else text[: THESIS_CHARS - 1].rstrip() + "…"


def _horizon(convictions: dict[str, dict]) -> str:
    counts = Counter(c.get("horizon") or "short-term" for c in convictions.values())
    top = max(counts.values())
    leaders = [h for h, n in counts.items() if n == top]
    return leaders[0] if len(leaders) == 1 else "short-term"


def compute_overall(convictions: dict[str, dict], base_weights: dict[str, float] | None = None) -> dict:
    """Combine ``{analyst name: conviction payload}`` into the overall conviction."""
    available = {
        name: c for name, c in convictions.items()
        if isinstance(c, dict) and isinstance(c.get("score"), (int, float))
    }
    if not available:
        raise NoConvictionError("no analyst produced a conviction")
    ordered = dict(sorted(
        available.items(),
        key=lambda kv: ANALYST_ORDER.index(kv[0]) if kv[0] in ANALYST_ORDER else len(ANALYST_ORDER),
    ))

    base = {**DEFAULT_WEIGHTS, **(base_weights or {})}
    raw = {
        name: max(float(base.get(name, 1 / 3)), 0.0) * max(float(c.get("confidence", 0.5)), _CONFIDENCE_FLOOR)
        for name, c in ordered.items()
    }
    total = sum(raw.values())
    if total <= 0:
        raw = dict.fromkeys(ordered, 1.0)
        total = float(len(ordered))
    weights = {name: w / total for name, w in raw.items()}

    scores = [float(c["score"]) for c in ordered.values()]
    score = sum(weights[name] * float(c["score"]) for name, c in ordered.items())
    score = int(round(max(1.0, min(100.0, score))))
    spread = pstdev(scores) if len(scores) > 1 else 0.0
    agreement = round(1.0 - min(1.0, spread / _AGREEMENT_SPREAD), 2)
    mean_confidence = sum(weights[name] * float(c.get("confidence", 0.5)) for name, c in ordered.items())
    confidence = round(mean_confidence * agreement, 2)

    return {
        "score": score,
        "label": conviction_label(score),
        "confidence": confidence,
        "agreement": agreement,
        "thesis": _thesis(ordered, agreement),
        "horizon": _horizon(ordered),
        "components": list(ordered),
        "weights": {name: round(w, 3) for name, w in weights.items()},
    }
