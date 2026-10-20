"""Tests for TRADINGAGENTS_* env-var overlay onto DEFAULT_CONFIG."""

from __future__ import annotations

import os

import pytest

import tradingagents.default_config as default_config_module


def _config_with_env(monkeypatch, **overrides):
    """The defaults as the environment given here would set them."""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return default_config_module.build_default_config()


def test_no_env_uses_built_in_defaults(monkeypatch):
    config = _config_with_env(monkeypatch)
    assert config["llm_provider"] == "openai"
    assert config["quick_think_llm"] == "gpt-6-luna"
    assert config["backend_url"] is None
    assert config["max_tool_rounds"] == 8
    assert config["llm_timeout"] == 90.0
    assert config["max_concurrent_runs"] == 2
    assert config["research_refresh_min_age_seconds"] == 0
    assert config["research_persist_results"] is True
    assert config["research_store_path"] is None


def test_pipeline_only_keys_are_gone(monkeypatch):
    config = _config_with_env(monkeypatch)
    for key in (
        "deep_think_llm", "results_dir", "memory_log_path", "checkpoint_enabled",
        "max_debate_rounds", "max_risk_discuss_rounds", "benchmark_map",
    ):
        assert key not in config


def test_string_overrides(monkeypatch):
    config = _config_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="google",
        TRADINGAGENTS_QUICK_THINK_LLM="gemini-3-flash-preview",
        TRADINGAGENTS_LLM_BACKEND_URL="https://example.invalid/v1",
        TRADINGAGENTS_OUTPUT_LANGUAGE="Chinese",
    )
    assert config["llm_provider"] == "google"
    assert config["quick_think_llm"] == "gemini-3-flash-preview"
    assert config["backend_url"] == "https://example.invalid/v1"
    assert config["output_language"] == "Chinese"


def test_numeric_coercion(monkeypatch):
    config = _config_with_env(
        monkeypatch,
        TRADINGAGENTS_MAX_TOOL_ROUNDS="5",
        TRADINGAGENTS_MAX_CONCURRENT_RUNS="3",
        TRADINGAGENTS_RESEARCH_REFRESH_MIN_AGE_SECONDS="60",
        TRADINGAGENTS_RESEARCH_PERSIST="false",
        TRADINGAGENTS_RESEARCH_STORE_PATH="/data/results.json",
        TRADINGAGENTS_LLM_TIMEOUT="45",
    )
    assert config["max_tool_rounds"] == 5 and isinstance(config["max_tool_rounds"], int)
    assert config["max_concurrent_runs"] == 3
    assert config["research_refresh_min_age_seconds"] == 60
    assert config["research_persist_results"] is False
    assert config["research_store_path"] == "/data/results.json"
    assert config["llm_timeout"] == 45.0 and isinstance(config["llm_timeout"], float)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ],
)
def test_bool_coercion(raw, expected):
    assert default_config_module._coerce(raw, False) is expected


@pytest.mark.parametrize("bad", ["treu", "flase", "maybe", "2", "enabled"])
def test_invalid_bool_raises(bad):
    """A misspelled boolean must fail loudly (like ints) instead of silently False."""
    with pytest.raises(ValueError, match="expected a boolean"):
        default_config_module._coerce(bad, True)


def test_reasoning_thinking_overrides(monkeypatch):
    """The provider reasoning/thinking knobs are env-configurable."""
    config = _config_with_env(
        monkeypatch,
        TRADINGAGENTS_OPENAI_REASONING_EFFORT="high",
        TRADINGAGENTS_GOOGLE_THINKING_LEVEL="minimal",
        TRADINGAGENTS_ANTHROPIC_EFFORT="low",
    )
    assert config["openai_reasoning_effort"] == "high"
    assert config["google_thinking_level"] == "minimal"
    assert config["anthropic_effort"] == "low"


def test_reasoning_effort_defaults_to_none(monkeypatch):
    """Unset reasoning/thinking knobs stay None so each provider uses its own default."""
    config = _config_with_env(monkeypatch)
    assert config["openai_reasoning_effort"] is None
    assert config["google_thinking_level"] is None
    assert config["anthropic_effort"] is None


def test_empty_env_value_is_passthrough(monkeypatch):
    """Empty TRADINGAGENTS_* values must not clobber the built-in default."""
    config = _config_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="",
        TRADINGAGENTS_MAX_TOOL_ROUNDS="",
    )
    assert config["llm_provider"] == "openai"
    assert config["max_tool_rounds"] == 8


def test_empty_path_value_keeps_the_default_path(monkeypatch):
    """.env.example lists the path variables blank; an empty value keeps the default."""
    config = _config_with_env(monkeypatch, TRADINGAGENTS_CACHE_DIR="")
    home = default_config_module._TRADINGAGENTS_HOME
    assert config["data_cache_dir"] == os.path.join(home, "cache")


def test_invalid_int_raises(monkeypatch):
    """Garbage int values should surface a ValueError at import, not silently misconfigure."""
    monkeypatch.setenv("TRADINGAGENTS_MAX_TOOL_ROUNDS", "not-a-number")
    with pytest.raises(ValueError, match="TRADINGAGENTS_MAX_TOOL_ROUNDS"):
        default_config_module.build_default_config()


def test_unknown_env_var_is_ignored(monkeypatch):
    """Env vars outside _ENV_OVERRIDES must not bleed into DEFAULT_CONFIG."""
    config = _config_with_env(
        monkeypatch,
        TRADINGAGENTS_NONEXISTENT_KEY="oops",
    )
    assert "nonexistent_key" not in config
