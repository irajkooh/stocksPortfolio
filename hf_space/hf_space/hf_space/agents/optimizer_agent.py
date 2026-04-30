"""LangGraph node: Markowitz optimizer."""
from __future__ import annotations
import logging
import re
from services.optimizer import optimize_portfolio, build_plots, save_allocation

log = logging.getLogger(__name__)

_BUDGET_RE  = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:k|K|thousand)?")
_PERCENT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
_MAX_SHARPE_RE = re.compile(r"\bmax(?:imum|imise|imize)?\s+sharpe\b", re.I)


def _extract_budget(msg: str) -> float | None:
    m = _BUDGET_RE.search(msg or "")
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        val = float(raw)
    except ValueError:
        return None
    if re.search(r"\d\s*(k|K|thousand)", msg):
        val *= 1000
    return val if val >= 1000 else None


def _extract_target_vol(msg: str) -> float | None:
    for m in _PERCENT_RE.finditer(msg or ""):
        v = float(m.group(1)) / 100
        if 0.02 <= v <= 0.60:
            return v
    return None


def _load_saved_params(portfolio_id: int) -> dict:
    try:
        from core.database import SessionLocal
        from core.models import PortfolioAllocationDB
        with SessionLocal() as s:
            row = s.get(PortfolioAllocationDB, portfolio_id)
            if row:
                return {"budget": row.budget, "target_vol": row.target_vol}
    except Exception:
        pass
    return {}


def optimizer_node(state: dict) -> dict:
    if "optimizer" not in state.get("active_agents", []):
        return state
    portfolio_id = state.get("active_portfolio_id", 1)
    # state["messages"][-1].content is the canonical message source
    msg = state["messages"][-1].content if state.get("messages") else ""

    budget = _extract_budget(msg)
    target_vol = _extract_target_vol(msg)

    # Fall back to saved allocation parameters when not specified in message
    if budget is None or target_vol is None:
        saved = _load_saved_params(portfolio_id)
        if budget is None:
            budget = saved.get("budget")
        if target_vol is None:
            if _MAX_SHARPE_RE.search(msg):
                # Max-Sharpe ≈ tangency portfolio; use a wide vol budget so the
                # optimizer is unconstrained and Sharpe-maximising weights emerge.
                target_vol = 0.30
            elif re.search(r"\bsharpe\b", msg, re.I):
                # "Sharpe of 1.5" etc. — not a direct vol target; use saved or default
                target_vol = saved.get("target_vol", 0.18)
            else:
                target_vol = saved.get("target_vol")

    if budget is None or target_vol is None:
        state["optimizer_result"] = {
            "error": (
                "To run the optimizer I need a budget and a target volatility, "
                "e.g. *'Rebalance with $50 000 at 18% volatility'*. "
                "You can also run the Optimizer tab directly to set these parameters."
            )
        }
        state.setdefault("agent_status", []).append(
            "🤖 Optimizer: missing budget/target_vol"
        )
        return state

    from core.database import SessionLocal
    from core.models import HoldingDB
    with SessionLocal() as s:
        tickers = [h.ticker for h in
                   s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()]

    try:
        result = optimize_portfolio(
            tickers=tickers, budget=budget, target_vol=target_vol,
            lookback="2y", risk_free_rate=0.04,
        )
        fig_pie, fig_bar, fig_frontier = build_plots(result)
        save_allocation(portfolio_id, result, budget=budget,
                        target_vol=target_vol, lookback="2y")
    except Exception as e:
        log.exception("optimizer failed")
        state["optimizer_result"] = {"error": str(e)}
        state.setdefault("agent_status", []).append(f"🤖 Optimizer failed: {e}")
        return state

    state["optimizer_result"] = result
    state.setdefault("charts", []).extend([fig_pie, fig_bar, fig_frontier])
    state.setdefault("agent_status", []).append(
        f"🤖 Optimizer: {len(result['allocations'])} tickers, "
        f"cash ${result['cash_dollars']:.0f}"
    )
    return state
