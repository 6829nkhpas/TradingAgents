import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_TOOL_ROUNDS":      "max_tool_rounds",
    "TRADINGAGENTS_TEMPERATURE":          "temperature",
    "TRADINGAGENTS_LLM_MAX_RETRIES":      "llm_max_retries",
    "TRADINGAGENTS_MAX_TOKENS":           "max_tokens",
    "TRADINGAGENTS_LLM_TIMEOUT":          "llm_timeout",
    "TRADINGAGENTS_MAX_CONCURRENT_RUNS":  "max_concurrent_runs",
    "TRADINGAGENTS_RESEARCH_REFRESH_MIN_AGE_SECONDS": "research_refresh_min_age_seconds",
    "TRADINGAGENTS_RESEARCH_PERSIST":     "research_persist_results",
    "TRADINGAGENTS_RESEARCH_STORE_PATH":  "research_store_path",
    # Provider-specific reasoning/thinking knobs (None = each provider's own default).
    "TRADINGAGENTS_GOOGLE_THINKING_LEVEL":   "google_thinking_level",
    "TRADINGAGENTS_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "TRADINGAGENTS_ANTHROPIC_EFFORT":        "anthropic_effort",
}


_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value.

    Invalid values raise ``ValueError`` rather than silently falling back to a
    default — a misspelled boolean (e.g. ``treu``) or non-numeric int should fail
    loudly at startup, not quietly misconfigure an unattended service.
    """
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(
            f"expected a boolean ({'/'.join(_BOOL_TRUE + _BOOL_FALSE)}), got {value!r}"
        )
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        try:
            config[key] = _coerce(raw, config.get(key))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}: {exc}") from exc
    return config


def build_default_config() -> dict:
    """The built-in defaults with the TRADINGAGENTS_* environment folded in.

    Read when the package is imported, as DEFAULT_CONFIG; call it again to see
    the environment as it is now.
    """
    return _apply_env_overrides({
        "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR") or os.path.join(_TRADINGAGENTS_HOME, "cache"),
        # LLM settings. Every analyst and scorer uses this one model.
        "llm_provider": "openai",
        "quick_think_llm": "gpt-6-luna",
        # When None, each provider's client falls back to its own default endpoint
        # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
        # Keeping a provider-specific URL here would leak (e.g. OpenAI's /v1 was
        # previously being forwarded to Gemini, producing malformed request URLs).
        "backend_url": None,
        # Provider-specific thinking configuration
        "google_thinking_level": None,      # "high", "minimal", etc.
        "openai_reasoning_effort": None,    # "medium", "high", "low"
        "anthropic_effort": None,           # "high", "medium", "low"
        # Sampling temperature, forwarded to every provider when set. None leaves
        # each provider at its own default. Lower values reduce run-to-run
        # variation on models that honor it; reasoning models largely ignore it.
        "temperature": None,
        # SDK retry budget forwarded to every provider chat client. None leaves each
        # provider/SDK at its own default (usually 2). Raise it to ride out bursty
        # 429 throttling on rate-limited deployments instead of failing a run (#1091).
        "llm_max_retries": None,
        # Cap on output tokens forwarded to every provider chat client. None leaves
        # each provider at its own default. Set it to bound a model that emits
        # unbounded reasoning/output and hangs or trips a gateway idle timeout
        # (e.g. some deepseek-v4-flash deployments, #1204).
        "max_tokens": None,
        # Per-request LLM timeout in seconds, forwarded to clients that accept it,
        # so one hung provider call cannot stall a research run indefinitely.
        "llm_timeout": 90.0,
        # Output language for analyst reports.
        "output_language": "English",
        "max_recur_limit": 100,
        # Rounds of tool calls an analyst may make before it is asked for its report.
        "max_tool_rounds": 8,
        # Research API (see research_store.py). Each symbol keeps its latest
        # result; a request is answered from it and queues a background refresh,
        # unless the result is younger than research_refresh_min_age_seconds
        # (0: every request refreshes) or a refresh is already queued or running.
        # At most max_concurrent_runs runs execute at once; the rest queue.
        "max_concurrent_runs": 2,
        "research_refresh_min_age_seconds": 0,
        # Keep the results in a file so a restart serves them at once. The path
        # defaults to <data_cache_dir>/research_results.json.
        "research_persist_results": True,
        "research_store_path": None,
        # Base weights for the overall conviction. Each analyst's weight is scaled
        # by its own confidence and renormalized over the analysts that scored.
        "conviction_weights": {
            "sentiment": 0.30,
            "news": 0.35,
            "fundamentals": 0.35,
        },
        # News / data fetching parameters
        # Increase for longer lookback strategies or to broaden macro coverage;
        # decrease to reduce token usage in agent prompts.
        "news_article_limit": 20,             # max articles per ticker (ticker-news)
        "global_news_article_limit": 10,      # max articles for global/macro news
        "global_news_lookback_days": 7,       # macro news lookback window
        # Search queries used by get_global_news for macro headlines. Extend or
        # replace to broaden geographic / sector coverage.
        "global_news_queries": [
            "Federal Reserve interest rates inflation",
            "S&P 500 earnings GDP economic outlook",
            "geopolitical risk trade war sanctions",
            "ECB Bank of England BOJ central bank policy",
            "oil commodities supply chain energy",
        ],
        # Used instead of global_news_queries when the instrument is a crypto asset.
        "crypto_global_news_queries": [
            "crypto regulation SEC CFTC",
            "bitcoin ETF flows",
            "stablecoin regulation",
            "Federal Reserve interest rates crypto market",
            "crypto exchange hack security",
        ],
        # Data vendor configuration
        # Category-level configuration (default for all tools in category).
        # The configured value is the exact vendor chain — requests are NOT silently
        # routed to vendors you didn't choose. For ordered fallback, list several,
        # e.g. "yfinance,alpha_vantage". "default" uses all available vendors.
        "data_vendors": {
            # Statements come from SEC EDGAR as filed (US filers), then Yahoo; the
            # overview and insider tools, which SEC EDGAR does not serve, from Yahoo.
            "fundamental_data": "sec_edgar,yfinance",  # Options: sec_edgar, alpha_vantage, yfinance
            "news_data": "yfinance",             # Options: alpha_vantage, yfinance
            "macro_data": "fred",                # Options: fred (needs FRED_API_KEY)
            "prediction_markets": "polymarket",  # Options: polymarket (keyless)
        },
        # Tool-level configuration (takes precedence over category-level)
        "tool_vendors": {
            # Example: "get_news": "alpha_vantage",  # Override category default
        },
    })


DEFAULT_CONFIG = build_default_config()
