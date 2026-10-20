"""What the analysts are actually told.

Two problems an audit found: one analyst's brief reached the model as a Python
tuple, and every analyst was asked for a trade call that nothing reads.
"""

from __future__ import annotations

import importlib

import pytest

ANALYSTS = ["sentiment_analyst", "news_analyst", "fundamentals_analyst"]


@pytest.mark.unit
@pytest.mark.parametrize("name", ANALYSTS)
def test_an_analyst_brief_is_text_not_a_python_object(name):
    """A trailing comma made one brief a tuple, so the model was handed its repr
    (quotes, parens and all) instead of the instruction."""
    import ast
    import inspect

    mod = importlib.import_module(f"tradingagents.agents.analysts.{name}")
    tree = ast.parse(inspect.getsource(mod))
    briefs = [node.value for node in ast.walk(tree)
              if isinstance(node, ast.Assign)
              and getattr(node.targets[0], "id", "") == "system_message"]
    assert briefs, f"{name} has no system_message"
    for brief in briefs:
        assert not isinstance(brief, ast.Tuple), "the brief is a tuple, not text"


@pytest.mark.unit
@pytest.mark.parametrize("name", ANALYSTS)
def test_an_analyst_is_not_asked_for_a_trade_call_nothing_reads(name):
    """The stop signal is never consumed, and asking for it makes an analyst
    open with a direction that then travels as evidence."""
    import inspect

    mod = importlib.import_module(f"tradingagents.agents.analysts.{name}")
    assert "FINAL TRANSACTION PROPOSAL" not in inspect.getsource(mod)
