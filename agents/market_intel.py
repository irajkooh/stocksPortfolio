"""Market Intelligence Agent — live prices, company info."""
import re
import logging
from agents.state import PortfolioAgentState
from services.stock_service import get_stock_info

logger = logging.getLogger(__name__)

# Common false-positives to ignore
_SKIP = {"I", "A", "AM", "PM", "US", "AI", "OR", "AT", "IN", "OF", "TO",
         "BE", "NO", "DO", "GO", "UP", "MY", "IT", "IS", "ON", "IF"}


def market_intel_node(state: PortfolioAgentState) -> dict:
    if "market_intel" not in state.get("active_agents", []):
        return {}

    message = state["messages"][-1].content

    # Extract 1–5 letter ALL-CAPS words as candidate tickers
    candidates = re.findall(r'\b[A-Z]{1,5}\b', message)
    tickers = [t for t in dict.fromkeys(candidates) if t not in _SKIP][:6]

    market_data: dict = {}
    for ticker in tickers:
        info = get_stock_info(ticker)
        if info.get("price", 0) > 0:
            market_data[ticker] = info

    status = (
        f"📈 Market Intel: fetched {len(market_data)} tickers"
        if market_data else "📈 Market Intel: no valid tickers found"
    )
    logger.info(status)
    return {
        "market_data":  market_data,
        "agent_status": state.get("agent_status", []) + [status],
    }
