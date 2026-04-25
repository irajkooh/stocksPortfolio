"""Portfolio Analyst Agent — watchlist view (tickers only, no positions tracked)."""
import logging
from agents.state import PortfolioAgentState
from core.database import SessionLocal
from core.models import HoldingDB
from services.stock_service import get_batch_prices, get_stock_info

logger = logging.getLogger(__name__)


def portfolio_analyst_node(state: PortfolioAgentState) -> dict:
    if "portfolio_analyst" not in state.get("active_agents", []):
        return {}

    portfolio_id = state.get("active_portfolio_id", 1)
    db = SessionLocal()
    try:
        holdings = (
            db.query(HoldingDB)
            .filter(HoldingDB.portfolio_id == portfolio_id)
            .all()
        )
    finally:
        db.close()

    if not holdings:
        return {
            "portfolio_data": {"empty": True, "message": "No holdings in portfolio."},
            "agent_status": state.get("agent_status", []) + ["💼 Portfolio: empty"],
        }

    prices = get_batch_prices([h.ticker for h in holdings])
    rows: list[dict] = []
    for h in holdings:
        price = prices.get(h.ticker, 0.0)
        info  = get_stock_info(h.ticker)
        rows.append({
            "ticker":        h.ticker,
            "name":          info.get("name", h.ticker),
            "current_price": round(price, 4),
            "sector":        info.get("sector", "Unknown"),
            "change_pct":    round(float(info.get("change_pct") or 0.0), 2),
        })

    portfolio_data = {
        "mode":       "watchlist",
        "note":       "Tickers only — no share counts or purchase prices tracked.",
        "n_holdings": len(rows),
        "holdings":   rows,
    }

    status = f"💼 Portfolio #{portfolio_id}: {len(rows)} tickers on watchlist"
    logger.info(status)
    return {
        "portfolio_data": portfolio_data,
        "agent_status":   state.get("agent_status", []) + [status],
    }
