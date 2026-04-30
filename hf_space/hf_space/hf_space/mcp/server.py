"""
FastMCP server — run locally to expose portfolio tools to Claude Desktop / MCP clients.

Usage:
    python mcp/server.py

Claude Desktop config (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "portfolio": {
          "command": "python",
          "args": ["/path/to/portfolio-app/mcp/server.py"]
        }
      }
    }
"""
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import FastMCP
from core.database import SessionLocal, init_db
from core.models import HoldingDB
from services.stock_service import get_stock_info, get_batch_prices
from services.rl_optimizer import optimize_portfolio

mcp = FastMCP("Portfolio Manager")


@mcp.tool()
def get_stock_price(ticker: str) -> dict:
    """Get current price and key info for a stock ticker."""
    return get_stock_info(ticker.upper())


@mcp.tool()
def list_portfolio() -> dict:
    """List all holdings in the portfolio with current prices and P&L."""
    init_db()
    db = SessionLocal()
    try:
        holdings = db.query(HoldingDB).all()
        if not holdings:
            return {"holdings": [], "total_value": 0, "total_pnl": 0}

        prices      = get_batch_prices([h.ticker for h in holdings])
        total_value = 0.0
        total_cost  = 0.0
        rows        = []

        for h in holdings:
            price  = prices.get(h.ticker, 0.0)
            shares = h.shares or 0.0
            purchase_price = h.purchase_price or 0.0
            value  = price * shares
            cost   = purchase_price * shares
            total_value += value
            total_cost  += cost
            rows.append({
                "ticker":         h.ticker,
                "shares":         shares,
                "purchase_price": purchase_price,
                "current_price":  round(price, 4),
                "value":          round(value, 2),
                "pnl":            round(value - cost, 2),
            })

        return {
            "holdings":    rows,
            "total_value": round(total_value, 2),
            "total_cost":  round(total_cost, 2),
            "total_pnl":   round(total_value - total_cost, 2),
        }
    finally:
        db.close()


@mcp.tool()
def optimize_weights(tickers: list[str], budget: float = 100_000.0,
                     period: str = "2y", timesteps: int = 5_000) -> dict:
    """
    Run RL portfolio optimisation and return optimal weights + dollar allocations.

    Parameters
    ----------
    tickers   : list of ticker symbols, e.g. ["AAPL", "MSFT", "GOOGL"]
    budget    : total investment budget in USD (default 100 000)
    period    : historical data window  ("1y", "2y", "3y", "5y")
    timesteps : PPO training steps (higher = slower but better)
    """
    result = optimize_portfolio(
        tickers   = [t.upper() for t in tickers],
        budget    = budget,
        period    = period,
        timesteps = timesteps,
    )
    if "error" in result:
        return result
    return {
        "weights":     result["weights"],
        "allocations": result["allocations"],
        "metrics":     result["metrics"],
        "budget":      result["budget"],
    }


if __name__ == "__main__":
    mcp.run()
