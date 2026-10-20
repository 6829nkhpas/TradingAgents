"""Crypto fundamentals analyst: tokenomics, valuation and network activity for a coin.

A crypto asset has no issuer financials, so the equity fundamentals analyst's
tools (statements, insider filings) have nothing to read. This analyst instead
fetches two live profiles before calling the model and puts them in the prompt:

  1. CoinGecko: supply and dilution, valuation (market cap, FDV), rank,
     developer activity and community size.
  2. Yahoo Finance: market cap, circulating supply and trading volume.

It makes one LLM call and binds no tools. Price action and technical analysis
are out of scope by design: this is a fundamentals read.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tradingagents.agents.analysts.turn import content_text
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS
from tradingagents.dataflows import evidence
from tradingagents.dataflows.errors import NoMarketDataError, VendorError
from tradingagents.dataflows.vendors.coingecko import fetch_coingecko_fundamentals, is_unavailable
from tradingagents.dataflows.vendors.yahoo.fundamentals import get_crypto_profile

logger = logging.getLogger(__name__)

COINGECKO_UNAVAILABLE_NOTICE = (
    "Crypto fundamentals used Yahoo Finance only; CoinGecko data was unavailable for this run."
)


def _yahoo_block(ticker: str, as_of_date: str) -> str:
    """Yahoo's crypto profile, or a placeholder; never raises into the graph."""
    try:
        return get_crypto_profile(ticker, as_of_date)
    except NoMarketDataError as exc:
        return f"<Yahoo Finance crypto profile unavailable: {exc.detail or 'no data'}>"
    except VendorError as exc:
        return f"<Yahoo Finance crypto profile unavailable: {exc}>"
    except Exception as exc:  # noqa: BLE001 — a profile failure must not end the analysis
        logger.warning("Yahoo crypto profile failed for %s: %s", ticker, exc)
        return f"<Yahoo Finance crypto profile unavailable: {type(exc).__name__}>"


def _system_message(ticker: str, as_of_date: str, coingecko_block: str, yahoo_block: str) -> str:
    return f"""You are a crypto fundamentals analyst. Write a fundamentals report for {ticker} as of {as_of_date} from the two data sources below, which have already been collected for you.

### CoinGecko
<start_of_coingecko>
{coingecko_block}
<end_of_coingecko>

### Yahoo Finance
<start_of_yahoo>
{yahoo_block}
<end_of_yahoo>

Cover, with the specific figures:
1. **Supply and dilution**: circulating vs. total and max supply, the FDV / market-cap gap, and what future issuance or unlocks imply for holders.
2. **Valuation and liquidity**: market cap and rank, trading volume relative to market cap, distance from the all-time high as valuation context.
3. **Network and developer activity**: commit cadence, contributors and repository activity as signs of a maintained project.
4. **Community and adoption**: community size and the categories/use cases the asset serves.
5. **Fundamental risks**: concentration, dilution, inactive development, thin liquidity, or missing data.

Rules: this is a fundamentals read, so make no price predictions and no technical or chart analysis. Use only the figures above; where a source is marked unavailable, say so and do not estimate it. A crypto asset has no company financials, so do not discuss earnings or balance sheets. End with a Markdown table of the key metrics and what each implies. {NO_EXTERNAL_TOOLS}{get_language_instruction()}"""


def create_crypto_fundamentals_analyst(llm):
    def crypto_fundamentals_analyst_node(state):
        ticker = state["company_of_interest"]
        as_of_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        coingecko_block = fetch_coingecko_fundamentals(ticker, as_of_date)
        if is_unavailable(coingecko_block):
            evidence.add_notice(COINGECKO_UNAVAILABLE_NOTICE)
        yahoo_block = _yahoo_block(ticker, as_of_date)

        messages = [
            SystemMessage(content=(
                "You are a helpful AI assistant, collaborating with other assistants. "
                "Report what your data supports; another agent decides the trade. "
                f"Today's date is {as_of_date}; treat it as 'now' for all analysis. "
                f"{instrument_context}\n"
                + _system_message(ticker, as_of_date, coingecko_block, yahoo_block)
            )),
            HumanMessage(content=f"Write the crypto fundamentals report for {ticker}."),
        ]
        result = llm.invoke(messages)
        report = content_text(result.content)
        return {
            "messages": [AIMessage(content=report)],
            "fundamentals_report": report,
        }

    return crypto_fundamentals_analyst_node
