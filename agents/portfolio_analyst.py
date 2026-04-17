"""Portfolio Analyst Agent — P&L, allocation, positions."""
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

    prices      = get_batch_prices([h.ticker for h in holdings])
    total_value = 0.0
    total_cost  = 0.0
    rows        = []

    for h in holdings:
        price = prices.get(h.ticker, 0.0)
        value = price * h.shares
        cost  = h.purchase_price * h.shares
        pnl   = value - cost
        total_value += value
        total_cost  += cost
        info = get_stock_info(h.ticker)
        rows.append({
            "ticker":         h.ticker,
            "name":           info.get("name", h.ticker),
            "shares":         h.shares,
            "purchase_price": h.purchase_price,
            "current_price":  round(price, 4),
            "value":          round(value, 2),
            "cost":           round(cost, 2),
            "pnl":            round(pnl, 2),
            "pnl_pct":        round(pnl / cost * 100, 2) if cost else 0.0,
            "weight_pct":     round(value / total_value * 100, 2) if total_value else 0.0,
            "sector":         info.get("sector", "Unknown"),
            "change_pct":     info.get("change_pct", 0.0),
        })

    pnl_total = total_value - total_cost
    portfolio_data = {
        "total_value":   round(total_value, 2),
        "total_cost":    round(total_cost, 2),
        "total_pnl":     round(pnl_total, 2),
        "total_pnl_pct": round(pnl_total / total_cost * 100, 2) if total_cost else 0.0,
        "n_holdings":    len(rows),
        "holdings":      rows,
    }

    status = (
        f"💼 Portfolio #{portfolio_id}: {len(rows)} holdings, "
        f"value=${total_value:,.0f}, P&L=${pnl_total:+,.0f}"
    )
    logger.info(status)
    return {
        "portfolio_data": portfolio_data,
        "agent_status":   state.get("agent_status", []) + [status],
    }
