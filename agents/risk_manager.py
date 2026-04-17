"""Risk Manager Agent — Sharpe, VaR, drawdown, Sortino, concentration."""
import logging
import numpy as np
from agents.state import PortfolioAgentState
from services.stock_service import get_historical

logger = logging.getLogger(__name__)


def risk_manager_node(state: PortfolioAgentState) -> dict:
    if "risk_manager" not in state.get("active_agents", []):
        return {}

    portfolio = state.get("portfolio_data", {})
    holdings  = portfolio.get("holdings", [])

    if not holdings:
        return {
            "risk_metrics": {"error": "No holdings to assess."},
            "agent_status": state.get("agent_status", []) + ["⚠️ Risk: no holdings"],
        }

    total_value = portfolio.get("total_value", 1.0) or 1.0
    returns_map: dict[str, np.ndarray] = {}

    for h in holdings:
        hist = get_historical(h["ticker"], period="1y")
        if not hist.empty:
            returns_map[h["ticker"]] = hist["Close"].pct_change().dropna().values

    if not returns_map:
        return {
            "risk_metrics": {"error": "Could not fetch historical data."},
            "agent_status": state.get("agent_status", []) + ["⚠️ Risk: data unavailable"],
        }

    # Value-weighted portfolio daily returns
    min_len = min(len(v) for v in returns_map.values())
    port_r  = np.zeros(min_len)
    for h in holdings:
        if h["ticker"] in returns_map:
            w       = h["value"] / total_value
            port_r += w * returns_map[h["ticker"]][-min_len:]

    ann_ret = port_r.mean() * 252
    ann_vol = port_r.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Sortino (downside only)
    neg = port_r[port_r < 0]
    sortino = ann_ret / (neg.std() * np.sqrt(252)) if len(neg) > 1 else 0.0

    # VaR & CVaR 95 %
    var95  = float(np.percentile(port_r, 5))
    cvar95 = float(port_r[port_r <= var95].mean()) if (port_r <= var95).any() else var95

    # Max drawdown
    cum        = (1 + port_r).cumprod()
    roll_max   = np.maximum.accumulate(cum)
    drawdowns  = (cum - roll_max) / roll_max
    max_dd     = float(drawdowns.min())

    # Concentration
    weights    = {h["ticker"]: h["value"] / total_value for h in holdings
                  if h["ticker"] in returns_map}
    hhi        = sum(w ** 2 for w in weights.values())
    max_weight = max(weights.values()) if weights else 0.0

    risk = {
        "sharpe_ratio":         round(sharpe, 3),
        "sortino_ratio":        round(sortino, 3),
        "annual_return_pct":    round(ann_ret * 100, 2),
        "annual_volatility_pct":round(ann_vol * 100, 2),
        "var_95_daily_pct":     round(var95  * 100, 2),
        "cvar_95_daily_pct":    round(cvar95 * 100, 2),
        "max_drawdown_pct":     round(max_dd * 100, 2),
        "max_single_weight_pct":round(max_weight * 100, 2),
        "concentration_hhi":    round(hhi, 4),
        "risk_level": (
            "🔴 HIGH"   if sharpe < 0.5 or max_dd < -0.20 else
            "🟡 MEDIUM" if sharpe < 1.0 else
            "🟢 LOW"
        ),
    }

    status = (f"🛡️ Risk: Sharpe={sharpe:.2f}, "
              f"MaxDD={max_dd*100:.1f}%, VaR95={var95*100:.1f}%")
    logger.info(status)
    return {
        "risk_metrics": risk,
        "agent_status": state.get("agent_status", []) + [status],
    }
