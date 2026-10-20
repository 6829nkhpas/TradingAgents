# TradingAgents Research API

A small FastAPI service that researches one stock or crypto asset and returns three
convictions (social sentiment, news, fundamentals) and one overall conviction, each
with the analyst's report and links to the sources behind it.

It does no technical (price or indicator) analysis, and it does not make or execute
trades. Output is research only, not financial advice.

## How it works

Three LLM analysts run in parallel for each request:

| Analyst | Sources |
|---|---|
| Social sentiment | StockTwits (Bullish/Bearish tagged messages), Reddit (finance subreddits for stocks; the coin's own and the crypto subreddits for crypto), Yahoo Finance news; optional TypeSafe post screening |
| News | Yahoo Finance (or Alpha Vantage) company and global news, FRED macro series, Polymarket prediction markets |
| Fundamentals | Stocks: SEC EDGAR statements, Yahoo Finance (or Alpha Vantage) profile and statements, insider transactions. Crypto: CoinGecko (supply, dilution, valuation, developer and community data) and Yahoo Finance |

Each finished report is scored by one structured LLM call into a conviction. The
overall conviction is computed in code from those, so it is reproducible:

- weight = base weight (`conviction_weights`: sentiment 0.30, news 0.35, fundamentals
  0.35) × the analyst's confidence, renormalised over the analysts that scored
- `score` = weighted mean of the analyst scores
- `agreement` = 1 − min(1, stdev(scores) / 25)
- `confidence` = weighted mean confidence × agreement

Scores run 1-100 with 50 neutral; labels are Bullish at 60+, Bearish at 40 or below,
Neutral between.

## Install and run

Requires Python 3.11+.

```bash
pip install .
cp .env.example .env        # then set your LLM provider key (and optional keys below)
tradingagents-api           # serves on TRADINGAGENTS_API_HOST:TRADINGAGENTS_API_PORT (default 0.0.0.0:8000)
```

For a local, non-Docker run set `TRADINGAGENTS_API_HOST=127.0.0.1`.

Docker:

```bash
docker compose up --build research-api      # published on 127.0.0.1:8000
```

### Environment

| Variable | Needed | Purpose |
|---|---|---|
| An LLM provider key (e.g. `OPENAI_API_KEY`) and `TRADINGAGENTS_LLM_PROVIDER` / `TRADINGAGENTS_QUICK_THINK_LLM` | yes | The model every analyst and scorer uses. Any provider in `tradingagents/llm_clients` works, including OpenAI-compatible relays and Ollama. |
| `COINGECKO_API_KEY` | for crypto | CoinGecko Demo key (header auth; `COINGECKO_API_PLAN=pro` for a Pro key). Without it crypto fundamentals use Yahoo Finance only. |
| `FRED_API_KEY` | optional | Macro series for the news analyst. |
| `SEC_EDGAR_USER_AGENT` | recommended | Contact string SEC requires for EDGAR requests. |
| `ALPHA_VANTAGE_API_KEY` | optional | Alternative news/fundamentals vendor. |
| `TYPESAFE_API_KEY` | optional | Screens off-topic social posts. |
| `TRADINGAGENTS_API_KEY` | optional | When set, requests need `Authorization: Bearer <key>`. |
| `TRADINGAGENTS_MAX_CONCURRENT_RUNS` | optional | Research runs executing at once; the rest queue. Default 2. |
| `TRADINGAGENTS_RESEARCH_REFRESH_MIN_AGE_SECONDS` | optional | Don't refresh a stored result younger than this. Default 0 (every request refreshes). |
| `TRADINGAGENTS_RESEARCH_PERSIST` / `TRADINGAGENTS_RESEARCH_STORE_PATH` | optional | Keep results in a JSON file (default `<cache dir>/research_results.json`) so a restart serves them. On by default. |
| `TRADINGAGENTS_LLM_TIMEOUT` | optional | Per-request LLM timeout in seconds; default 90. |
| `TRADINGAGENTS_MAX_TOOL_ROUNDS` | optional | Tool-call rounds per analyst before it must report; default 8. |

**Security:** the API has no authentication unless `TRADINGAGENTS_API_KEY` is set.
Keep it on a private network (or behind a firewall allowlist), or set the key, before
exposing it; every request spends LLM credits.

## API

### `GET /health`

`{"status": "ok", "service": "TradingAgents Research API"}`. Calls no LLM or vendor.

### `POST /api/v1/research`

Request:

```json
{"symbol": "BTC", "asset_type": "crypto", "analysis_date": "2026-10-17"}
```

`asset_type` is optional for common coins (`BTC-USD`, `Bitcoin`) and needed for a
less common or ambiguous bare ticker. `analysis_date` defaults to today and cannot be
in the future.

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/research \
  -H "Content-Type: application/json" -d '{"symbol": "AAPL"}'

curl -s -X POST http://127.0.0.1:8000/api/v1/research \
  -H "Content-Type: application/json" -d '{"symbol": "BTC", "asset_type": "crypto"}'
```

Response (fields that do not apply are omitted; there are never `null`s):

```json
{
  "symbol": "BTC-USD",
  "base_asset": "BTC",
  "asset_type": "crypto",
  "analysis_date": "2026-10-17",
  "generated_at": "2026-10-17T10:00:00Z",
  "cached": true,
  "refreshing": true,
  "overall": {
    "score": 57, "label": "Neutral", "confidence": 0.37, "agreement": 0.62,
    "thesis": "Sentiment Bullish 66, News Bullish 61, Fundamentals Neutral 44; analysts mixed.",
    "horizon": "short-term",
    "components": ["sentiment", "news", "fundamentals"],
    "weights": {"sentiment": 0.3, "news": 0.35, "fundamentals": 0.35}
  },
  "analysts": {
    "sentiment": {
      "status": "ok",
      "conviction": {
        "score": 66, "label": "Bullish", "confidence": 0.6,
        "thesis": "Retail flow is upbeat on ETF inflows.",
        "drivers": [{"headline": "StockTwits 72% bullish", "impact": "positive", "weight": 0.4}],
        "risks": ["Thin sample on Reddit"],
        "horizon": "short-term"
      },
      "report": "**Overall Sentiment:** ...",
      "evidence": [
        {"id": "e1", "type": "social_post", "title": "...", "url": "https://stocktwits.com/...",
         "source": "StockTwits", "published_at": "2026-10-17T09:12:00Z", "stance": "Bullish"}
      ],
      "sources": ["Yahoo Finance", "StockTwits", "Reddit"]
    },
    "news": {"status": "ok", "...": "same shape"},
    "fundamentals": {"status": "unavailable", "reason": "The Fundamentals Analyst failed (TimeoutError).",
                     "sources": ["CoinGecko", "Yahoo Finance"]}
  },
  "notices": ["Configured sources may be unavailable or rate-limited; see each report for data gaps."],
  "disclaimer": "Research and analytics only; not financial or investment advice."
}
```

Analyst `status`:

- `ok`: `report` and `conviction`
- `degraded`: `report`, but it could not be scored (no `conviction`)
- `unavailable`: no report; `reason` says why

`overall` is omitted (with a notice) when no report could be scored. Evidence `type`
is one of `article`, `social_post`, `prediction_market`, `data`; every link comes from
the vendor's own data, never from the model.

Errors: `422` for an invalid symbol, asset type or date; `401` for a missing or wrong
bearer key (only when one is configured); `502` when every analyst failed or a provider
error prevented a response; `503` when the LLM provider is not configured.

### Latency and caching

A fresh run makes several LLM calls per analyst, so it takes from under a minute to
several minutes depending on the model. To keep requests fast, every symbol keeps its
latest result:

- A request for a symbol that has a stored result gets it straight away (`cached: true`).
- That request also queues a refresh for the symbol (`refreshing: true`), unless one
  is already queued or running. When it finishes it replaces the stored result, so the
  next request gets the newer one. Any number of requests for the same symbol share one
  refresh, and none of them waits for it.
- Only a symbol never researched before waits for its first run; concurrent requests
  for it share that run.
- Runs go through a queue with `TRADINGAGENTS_MAX_CONCURRENT_RUNS` workers. A failed
  refresh keeps the previous result.
- Results are saved to disk, so a restart serves them at once. In Docker they live in
  the `tradingagents_data` volume.

A request for today shares the symbol's latest result whatever day it was computed;
`analysis_date` and `generated_at` say when. A past `analysis_date` is its own entry,
computed once and never refreshed.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

Tests never reach the network (`tests/conftest.py` blocks it); vendors and models are
faked.

## License

Apache 2.0; see `LICENSE`. Built on [TradingAgents](https://github.com/TauricResearch/TradingAgents)
by Tauric Research.
