"""The research API loads only the research code.

Importing the API must not drag in the trading pipeline (debate, trader, risk
team, portfolio manager, memory, reporting, checkpointing) or the technical
analysis stack. Checked in a fresh interpreter so other tests' imports do not
mask a regression.
"""

import json
import subprocess
import sys

import pytest

FORBIDDEN_MODULES = (
    # Trading pipeline
    "tradingagents.graph.trading_graph",
    "tradingagents.graph.propagation",
    "tradingagents.graph.conditional_logic",
    "tradingagents.graph.checkpointer",
    "tradingagents.memory",
    "tradingagents.reporting",
    "tradingagents.backtest",
    "tradingagents.portfolio",
    "tradingagents.agents.rating",
    "tradingagents.agents.trader",
    "tradingagents.agents.researchers",
    "tradingagents.agents.managers",
    "tradingagents.agents.risk_mgmt",
    "langgraph.checkpoint.sqlite",
    # Technical / market analysis
    "stockstats",
    "tradingagents.agents.analysts.market_analyst",
    "tradingagents.dataflows.vendors.yahoo.market",
    "tradingagents.dataflows.vendors.yahoo.snapshot",
    "tradingagents.dataflows.vendors.yahoo.ohlcv",
    "tradingagents.dataflows.vendors.alpha_vantage.stock",
    "tradingagents.dataflows.vendors.alpha_vantage.indicator",
)

_PROBE = (
    "import json, sys\n"
    "import tradingagents.api.app\n"
    "print(json.dumps(sorted(sys.modules)))\n"
)


@pytest.mark.unit
def test_api_import_loads_only_research_code():
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    offenders = sorted(
        name
        for name in loaded
        if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN_MODULES)
    )
    assert offenders == []
