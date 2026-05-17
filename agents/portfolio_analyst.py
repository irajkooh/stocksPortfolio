"""Portfolio Analyst Agent — optimization winners when available, else positions/watchlist."""
import json
import logging
from agents.state import PortfolioAgentState
from core.database import SessionLocal
from core.models import HoldingDB, PortfolioAllocationDB
from services.stock_service import get_batch_prices, get_stock_info

logger = logging.getLogger(__name__)


def portfolio_analyst_node(state: PortfolioAgentState) -> dict:
    if "portfolio_analyst" not in state.get("active_agents", []):
        return {}

    portfolio_id = state.get("active_portfolio_id", 1)
    with SessionLocal() as s:
        all_holdings = s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()
        alloc_row    = s.get(PortfolioAllocationDB, portfolio_id)

    if not all_holdings:
        return {
            "portfolio_data": {"empty": True, "message": "No holdings in portfolio."},
            "agent_status": state.get("agent_status", []) + ["💼 Portfolio: empty"],
        }

    # ── Priority: optimization winners → bought → full watchlist ──────────────
    if alloc_row:
        alloc_data  = json.loads(alloc_row.allocations_json)
        opt_tickers = set(alloc_data.keys())
        active      = [h for h in all_holdings if h.ticker in opt_tickers]
        mode        = "optimized"
    else:
        bought  = [h for h in all_holdings if h.shares is not None and h.shares > 0]
        active  = bought if bought else all_holdings
        mode    = "positions" if bought else "watchlist"

    tickers = [h.ticker for h in active]
    prices  = get_batch_prices(tickers)

    rows: list[dict] = []
    total_market_value = 0.0
    total_today_pnl    = 0.0
    total_cost_basis   = 0.0
    total_pnl          = 0.0

    for h in active:
        price      = prices.get(h.ticker, 0.0)
        info       = get_stock_info(h.ticker)
        change_pct = float(info.get("change_pct") or 0.0)

        row: dict = {
            "ticker":        h.ticker,
            "name":          info.get("name", h.ticker),
            "current_price": round(price, 4),
            "sector":        info.get("sector", "Unknown"),
            "change_pct_1d": round(change_pct, 2),
        }

        # Attach optimizer allocation if available
        if alloc_row and h.ticker in alloc_data:
            v = alloc_data[h.ticker]
            row["opt_weight"]  = round(float(v["weight"]) * 100, 2)   # percent
            row["opt_dollars"] = round(float(v["dollars"]), 2)
            row["opt_shares"]  = round(float(v["shares"]), 4)

            shares       = float(v["shares"])
            market_value = shares * price
            if abs(1 + change_pct / 100) > 1e-9:
                prev_close = price / (1 + change_pct / 100)
            else:
                prev_close = price
            today_pnl = shares * (price - prev_close)

            row["market_value"] = round(market_value, 2)
            row["today_pnl"]    = round(today_pnl, 2)
            total_market_value += market_value
            total_today_pnl    += today_pnl

            opt_price = float(v.get("price", 0.0))
            if opt_price > 0:
                cost_basis  = shares * opt_price
                unrealized  = market_value - cost_basis
                total_pnl      += unrealized
                total_cost_basis += cost_basis
                row["opt_purchase_price"] = round(opt_price, 4)
                row["cost_basis"]         = round(cost_basis, 2)
                row["unrealized_pnl"]     = round(unrealized, 2)
                row["unrealized_pnl_pct"] = round(unrealized / cost_basis * 100, 2)

        elif mode == "positions" and h.shares and h.shares > 0:
            shares       = h.shares
            market_value = shares * price
            if abs(1 + change_pct / 100) > 1e-9:
                prev_close = price / (1 + change_pct / 100)
            else:
                prev_close = price
            today_pnl = shares * (price - prev_close)

            row["shares"]       = shares
            row["market_value"] = round(market_value, 2)
            row["today_pnl"]    = round(today_pnl, 2)
            total_market_value += market_value
            total_today_pnl    += today_pnl

            if h.purchase_price and h.purchase_price > 0:
                cost_basis  = shares * h.purchase_price
                unrealized  = market_value - cost_basis
                total_pnl      += unrealized
                total_cost_basis += cost_basis
                row["purchase_price"]     = round(h.purchase_price, 4)
                row["cost_basis"]         = round(cost_basis, 2)
                row["unrealized_pnl"]     = round(unrealized, 2)
                row["unrealized_pnl_pct"] = round(unrealized / cost_basis * 100, 2)

        rows.append(row)

    portfolio_data: dict = {
        "mode":       mode,
        "n_holdings": len(rows),
        "holdings":   rows,
    }

    if mode == "optimized":
        portfolio_data["note"] = (
            "Holdings are OPTIMIZATION WINNERS — tickers the optimizer selected. "
            "Non-winners were excluded by the Sharpe ratio screen."
        )
        portfolio_data["opt_budget"]          = alloc_row.budget
        portfolio_data["opt_expected_return"] = f"{alloc_row.expected_return*100:.2f}%"
        portfolio_data["opt_expected_vol"]    = f"{alloc_row.expected_vol*100:.2f}%"
        portfolio_data["opt_sharpe"]          = round(alloc_row.sharpe, 3)
        portfolio_data["opt_sortino"]         = round(alloc_row.sortino or 0.0, 3)
        portfolio_data["opt_cash_dollars"]    = alloc_row.cash_dollars
        portfolio_data["opt_last_run"]        = str(alloc_row.created_at)
        if total_market_value:
            portfolio_data["total_market_value"] = round(total_market_value, 2)
            portfolio_data["today_pnl"]           = round(total_today_pnl, 2)
        if total_cost_basis:
            portfolio_data["total_cost_basis"]        = round(total_cost_basis, 2)
            portfolio_data["total_unrealized_pnl"]    = round(total_pnl, 2)
            portfolio_data["total_unrealized_pnl_pct"] = round(total_pnl / total_cost_basis * 100, 2)
    elif mode == "positions":
        if total_market_value:
            portfolio_data["total_market_value"] = round(total_market_value, 2)
            portfolio_data["today_pnl"]           = round(total_today_pnl, 2)
        if total_cost_basis:
            portfolio_data["total_cost_basis"]        = round(total_cost_basis, 2)
            portfolio_data["total_unrealized_pnl"]    = round(total_pnl, 2)
            portfolio_data["total_unrealized_pnl_pct"] = round(total_pnl / total_cost_basis * 100, 2)
    else:
        portfolio_data["note"] = (
            "WATCHLIST ONLY. No share counts or purchase prices are stored. "
            "P&L, portfolio value, cost basis, and returns CANNOT be calculated. "
            "Do NOT estimate or invent these values."
        )

    status = f"💼 Portfolio #{portfolio_id}: {len(rows)} {mode} holdings"
    if total_today_pnl:
        status += f", today P&L ${total_today_pnl:+,.2f}"
    logger.info(status)
    return {
        "portfolio_data": portfolio_data,
        "agent_status":   state.get("agent_status", []) + [status],
    }
