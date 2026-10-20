"""CoinGecko crypto fundamentals: supply, dilution, valuation, network and community.

Used by the crypto fundamentals analyst. Reads ``GET /coins/{id}`` from the
CoinGecko API with the key in a request header (never the URL, so it cannot
reach logs or error messages):

- Demo plan (default): ``https://api.coingecko.com/api/v3`` with ``x-cg-demo-api-key``.
- Pro plan (``COINGECKO_API_PLAN=pro``): ``https://pro-api.coingecko.com/api/v3``
  with ``x-cg-pro-api-key``.
- ``COINGECKO_API_BASE_URL`` overrides the root.

Deliberately no price statistics (price changes, highs/lows as a chart signal,
indicators): this is a fundamentals read. "Distance below the all-time high"
is kept as a valuation-context figure.

Like the other vendors it returns a formatted text block and never raises into
the graph: a missing key, a rejected key, a rate limit or a network error
becomes a ``<CoinGecko unavailable: ...>`` placeholder.
"""

from __future__ import annotations

import logging
import os
import threading

import requests

from tradingagents.dataflows import evidence
from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.symbols import crypto_base

logger = logging.getLogger(__name__)

DEMO_ROOT = "https://api.coingecko.com/api/v3"
PRO_ROOT = "https://pro-api.coingecko.com/api/v3"
REQUEST_TIMEOUT = 10
PLACEHOLDER_PREFIX = "<CoinGecko unavailable"

# Base asset -> CoinGecko coin id for the common coins, so they need no search.
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "BNB": "binancecoin",
    "TRX": "tron",
    "TON": "the-open-network",
    "SHIB": "shiba-inu",
    "MATIC": "matic-network",
    "POL": "polygon-ecosystem-token",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "XLM": "stellar",
    "NEAR": "near",
    "APT": "aptos",
    "ARB": "arbitrum",
    "OP": "optimism",
    "SUI": "sui",
    "PEPE": "pepe",
    "FIL": "filecoin",
    "ICP": "internet-computer",
    "ETC": "ethereum-classic",
    "HBAR": "hedera-hashgraph",
    "INJ": "injective-protocol",
    "AAVE": "aave",
    "TIA": "celestia",
    "SEI": "sei-network",
    "WIF": "dogwifcoin",
    "BONK": "bonk",
    "USDT": "tether",
    "USDC": "usd-coin",
}

_resolved_ids: dict[str, str | None] = {}
_ids_lock = threading.Lock()


class CoinGeckoError(Exception):
    """A CoinGecko request failed; the message is safe to show (holds no key)."""


def _api() -> tuple[str, dict[str, str]]:
    """API root and auth header from the environment; raises when no key is set."""
    key = (os.getenv("COINGECKO_API_KEY") or "").strip()
    if not key:
        raise CoinGeckoError("COINGECKO_API_KEY is not set")
    pro = (os.getenv("COINGECKO_API_PLAN") or "demo").strip().lower() == "pro"
    root = (os.getenv("COINGECKO_API_BASE_URL") or (PRO_ROOT if pro else DEMO_ROOT)).rstrip("/")
    header = "x-cg-pro-api-key" if pro else "x-cg-demo-api-key"
    return root, {header: key, "Accept": "application/json"}


def _get(path: str, params: dict) -> dict:
    root, headers = _api()
    try:
        response = requests.get(f"{root}/{path}", params=params, headers=headers,
                                timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        # The URL holds no key, but keep the message to the error class anyway.
        raise CoinGeckoError(f"network error ({type(exc).__name__})") from None
    if response.status_code in (401, 403):
        raise CoinGeckoError(f"API key rejected (HTTP {response.status_code})")
    if response.status_code == 429:
        raise CoinGeckoError("rate limited (HTTP 429)")
    if response.status_code == 404:
        raise CoinGeckoError("coin not found (HTTP 404)")
    if response.status_code >= 400:
        raise CoinGeckoError(f"HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        raise CoinGeckoError("invalid JSON response") from None
    if not isinstance(data, dict):
        raise CoinGeckoError("unexpected response shape")
    return data


def coin_base(ticker: str) -> str:
    """The base asset of a crypto ticker: ``BTC-USD`` -> ``BTC``, ``SHIB`` -> ``SHIB``."""
    return crypto_base(ticker, any_base=True) or ticker.strip().upper().split("-")[0]


def resolve_coin_id(base: str) -> str | None:
    """CoinGecko's coin id for ``base``: the built-in table, else ``/search``.

    A search answer (including "no match") is cached for the process; a failed
    search is not, so the next run asks again.
    """
    base = base.strip().upper()
    if base in COINGECKO_IDS:
        return COINGECKO_IDS[base]
    with _ids_lock:
        if base in _resolved_ids:
            return _resolved_ids[base]
    data = _get("search", {"query": base})
    coins = data.get("coins") or []
    exact = [c for c in coins if str(c.get("symbol", "")).upper() == base and c.get("id")]
    # Search results are ranked by market cap, so the first exact symbol match
    # is the coin people mean by that ticker.
    coin_id = exact[0]["id"] if exact else None
    with _ids_lock:
        _resolved_ids[base] = coin_id
    return coin_id


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _usd(market: dict, key: str) -> float | None:
    value = market.get(key)
    return _num(value.get("usd")) if isinstance(value, dict) else _num(value)


def _fmt(value: float | None, unit: str = "") -> str | None:
    if value is None:
        return None
    if float(value).is_integer():
        text = f"{int(value):,}"
    elif abs(value) >= 1000:
        text = f"{value:,.0f}"
    elif abs(value) >= 1:
        text = f"{value:,.2f}"
    else:
        text = f"{value:.6g}"
    return f"{unit}{text}"


def format_coin(coin: dict) -> str:
    """Render a ``/coins/{id}`` payload as the analyst's fundamentals block."""
    market = coin.get("market_data") or {}
    dev = coin.get("developer_data") or {}
    community = coin.get("community_data") or {}

    market_cap = _usd(market, "market_cap")
    fdv = _usd(market, "fully_diluted_valuation")
    circulating = _num(market.get("circulating_supply"))
    total = _num(market.get("total_supply"))
    max_supply = _num(market.get("max_supply"))
    ath_change = market.get("ath_change_percentage")
    ath_change_usd = _num(ath_change.get("usd")) if isinstance(ath_change, dict) else None

    fdv_ratio = fdv / market_cap if fdv and market_cap else None
    supply_basis = max_supply or total
    issued = circulating / supply_basis if circulating and supply_basis else None
    categories = [c for c in (coin.get("categories") or []) if c][:6]

    sections = {
        "Identity": [
            ("Name", coin.get("name")),
            ("Symbol", str(coin.get("symbol") or "").upper() or None),
            ("Market Cap Rank", coin.get("market_cap_rank")),
            ("Categories", ", ".join(categories) or None),
            ("Genesis Date", coin.get("genesis_date")),
            ("Hashing Algorithm", coin.get("hashing_algorithm")),
        ],
        "Valuation": [
            ("Market Cap", _fmt(market_cap, "$")),
            ("Fully Diluted Valuation", _fmt(fdv, "$")),
            ("FDV / Market Cap", f"{fdv_ratio:.2f}x" if fdv_ratio else None),
            ("24h Trading Volume (liquidity)", _fmt(_usd(market, "total_volume"), "$")),
            ("Below All-Time High", f"{abs(ath_change_usd):.1f}%" if ath_change_usd is not None
             and ath_change_usd < 0 else None),
        ],
        "Supply & Dilution": [
            ("Circulating Supply", _fmt(circulating)),
            ("Total Supply", _fmt(total)),
            ("Max Supply", _fmt(max_supply) if max_supply else "uncapped / not set"),
            ("Circulating / Max (or Total)", f"{issued:.1%}" if issued else None),
        ],
        "Developer Activity": [
            ("Commits (last 4 weeks)", _fmt(_num(dev.get("commit_count_4_weeks")))),
            ("GitHub Stars", _fmt(_num(dev.get("stars")))),
            ("Forks", _fmt(_num(dev.get("forks")))),
            ("Merged Pull Requests", _fmt(_num(dev.get("pull_requests_merged")))),
            ("Contributors", _fmt(_num(dev.get("pull_request_contributors")))),
        ],
        "Community": [
            ("Reddit Subscribers", _fmt(_num(community.get("reddit_subscribers")))),
            ("Telegram Members", _fmt(_num(community.get("telegram_channel_user_count")))),
            ("Sentiment Votes Up", f"{coin['sentiment_votes_up_percentage']}%"
             if coin.get("sentiment_votes_up_percentage") is not None else None),
        ],
    }

    lines = [f"# CoinGecko fundamentals for {coin.get('name') or coin.get('id')}"]
    for title, fields in sections.items():
        present = [(label, value) for label, value in fields if value is not None and value != ""]
        if present:
            lines.append(f"\n## {title}")
            lines.extend(f"- {label}: {value}" for label, value in present)
    return "\n".join(lines)


def fetch_coingecko_fundamentals(ticker: str, as_of_date: str | None = None) -> str:
    """The CoinGecko fundamentals block for a crypto ``ticker``, or a placeholder."""
    if is_historical(as_of_date):
        return (
            f"{PLACEHOLDER_PREFIX}: withheld for {as_of_date}. CoinGecko serves a live "
            f"snapshot with no historical vintage, so it cannot describe that date.>"
        )
    base = coin_base(ticker)
    try:
        coin_id = resolve_coin_id(base)
        if not coin_id:
            return f"{PLACEHOLDER_PREFIX}: no CoinGecko coin matches the symbol {base}>"
        coin = _get(f"coins/{coin_id}", {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false",
        })
    except CoinGeckoError as exc:
        logger.warning("CoinGecko fetch failed for %s: %s", base, exc)
        return f"{PLACEHOLDER_PREFIX}: {exc}>"

    evidence.record(
        type="data",
        title=f"{coin.get('name') or base} on CoinGecko",
        source="CoinGecko",
        url=f"https://www.coingecko.com/en/coins/{coin_id}",
    )
    return format_coin(coin)


def is_unavailable(block: str) -> bool:
    """Whether ``fetch_coingecko_fundamentals`` returned a placeholder."""
    return block.startswith(PLACEHOLDER_PREFIX)
