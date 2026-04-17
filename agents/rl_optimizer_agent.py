"""
RL Optimizer Agent
------------------
Auto-triggered by supervisor when rebalancing / optimisation intent is detected.
Reads tickers from the active portfolio (or from the message) and budget from
the message, then runs PPO optimisation.
"""
import logging
import re
from agents.state import PortfolioAgentState
from services.rl_optimizer import optimize_portfolio, build_plots
from core.database import SessionLocal
from core.models import HoldingDB

logger = logging.getLogger(__name__)

_BUDGET_RE = re.compile(
    r'\$?\s*([\d,]+(?:\.\d+)?)\s*([kKmM]?)\s*(?:dollars?|usd|budget)?',
    re.IGNORECASE,
)


def _extract_budget(message: str) -> float:
    """Parse a dollar amount from free text. Returns 100 000 as default."""
    for m in _BUDGET_RE.finditer(message):
        raw    = float(m.group(1).replace(",", ""))
        suffix = m.group(2).lower()
        if suffix == "k":
            raw *= 1_000
        elif suffix == "m":
            raw *= 1_000_000
        if raw >= 100:
            return raw
    return 100_000.0


def _portfolio_tickers(portfolio_id: int = 1) -> list[str]:
    db = SessionLocal()
    try:
        return [
            h.ticker
            for h in db.query(HoldingDB)
            .filter(HoldingDB.portfolio_id == portfolio_id)
            .all()
        ]
    finally:
        db.close()


def rl_optimizer_node(state: PortfolioAgentState) -> dict:
    if "rl_optimizer" not in state.get("active_agents", []):
        return {}

    portfolio_id = state.get("active_portfolio_id", 1)
    message = state["messages"][-1].content
    budget  = _extract_budget(message)

    # Prefer portfolio tickers; fall back to tickers mentioned in message
    tickers = _portfolio_tickers(portfolio_id)
    if len(tickers) < 2:
        candidates = re.findall(r'\b[A-Z]{1,5}\b', message)
        skip = {"I", "A", "AM", "PM", "US", "AI", "OR", "AT", "IN", "OF", "TO", "BE"}
        tickers = [t for t in dict.fromkeys(candidates) if t not in skip]

    if len(tickers) < 2:
        return {
            "rl_result": {"error": "Need ≥ 2 tickers to optimise."},
            "agent_status": state.get("agent_status", []) + [
                "🤖 RL Optimizer: insufficient tickers"
            ],
        }

    status_running = (
        f"🤖 RL Optimizer: training PPO on {tickers} — budget ${budget:,.0f} …"
    )
    logger.info(status_running)

    result = optimize_portfolio(
        tickers   = tickers,
        budget    = budget,
        period    = "2y",
        timesteps = 10_000,
    )

    if "error" in result:
        return {
            "rl_result":    result,
            "agent_status": state.get("agent_status", []) + [
                f"🤖 RL Optimizer: {result['error']}"
            ],
        }

    fig_w, fig_p, fig_f, fig_b = build_plots(result)

    serialisable = {
        k: v for k, v in result.items()
        if k not in ("returns_df", "prices_df", "final_weights")
    }

    status_done = (
        f"🤖 RL Optimizer: done — "
        f"Sharpe {result['metrics']['rl_sharpe']:.3f} "
        f"(vs {result['metrics']['eq_sharpe']:.3f} equal-weight)"
    )
    logger.info(status_done)

    return {
        "rl_result":    serialisable,
        "charts":       state.get("charts", []) + [fig_w, fig_p, fig_f, fig_b],
        "agent_status": state.get("agent_status", []) + [status_done],
    }
