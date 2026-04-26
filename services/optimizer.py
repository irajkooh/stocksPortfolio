"""Markowitz mean-variance optimizer with always-on risk-free CASH asset."""
from __future__ import annotations
import json
import logging
from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

from services.stock_service import get_historical, get_stock_info

log = logging.getLogger(__name__)

TRADING_DAYS = 252
MIN_TRADING_DAYS = 60
CASH = "CASH"


def _collect_returns(tickers: list[str], lookback: str) -> pd.DataFrame:
    frames = {}
    for t in tickers:
        df = get_historical(t, period=lookback)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        frames[t] = df["Close"]
    if len(frames) < 3:
        raise ValueError(f"need at least 3 tickers with price history, got {len(frames)}")
    prices = pd.concat(frames.values(), axis=1, keys=frames.keys()).dropna()
    if len(prices) < MIN_TRADING_DAYS:
        raise ValueError(
            f"not enough history: {len(prices)} rows < {MIN_TRADING_DAYS}"
        )
    return prices.pct_change().dropna()


def _solve_max_return(mu: np.ndarray, cov: np.ndarray, target_var: float,
                      max_w_risky: float, n_risky: int) -> tuple[np.ndarray, bool]:
    """Solve: maximize wᵀμ  s.t. wᵀΣw ≤ target_var, Σw = 1, bounds."""
    n = len(mu)  # n_risky + 1 (cash is last)
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, max_w_risky)] * n_risky + [(0.0, 1.0)]  # cash unbounded up to 1
    cons = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: target_var - w @ cov @ w},
    ]
    res = minimize(lambda w: -float(w @ mu), x0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-9, "maxiter": 300})
    return res.x, res.success


def _solve_min_var(mu: np.ndarray, cov: np.ndarray,
                   max_w_risky: float, n_risky: int) -> np.ndarray:
    n = len(mu)
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, max_w_risky)] * n_risky + [(0.0, 1.0)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(lambda w: float(w @ cov @ w), x0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-9, "maxiter": 300})
    return res.x


def optimize_portfolio(
    tickers: list[str],
    budget: float,
    target_vol: float,
    lookback: str = "2y",
    risk_free_rate: float = 0.04,
    max_weight: float = 0.40,
    frontier_samples: int = 120,
) -> dict[str, Any]:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if len(set(tickers)) < 3:
        raise ValueError("need at least 3 unique tickers")

    returns_df = _collect_returns(tickers, lookback)
    used = list(returns_df.columns)

    mu_annual  = returns_df.mean().values * TRADING_DAYS
    cov_annual = returns_df.cov().values * TRADING_DAYS
    n_risky = len(used)

    mu_aug  = np.concatenate([mu_annual, [risk_free_rate]])
    cov_aug = np.zeros((n_risky + 1, n_risky + 1))
    cov_aug[:n_risky, :n_risky] = cov_annual

    warnings: list[str] = []
    target_var = target_vol ** 2

    w_star, ok = _solve_max_return(mu_aug, cov_aug, target_var,
                                   max_weight, n_risky)
    realized_vol = float(np.sqrt(w_star @ cov_aug @ w_star))
    if (not ok) or realized_vol > target_vol + 1e-3 or w_star[-1] > 0.99:
        w_star = _solve_min_var(mu_aug, cov_aug, max_weight, n_risky)
        warnings.append(
            f"target_vol={target_vol:.3f} infeasible; fell back to min-var portfolio"
        )

    w_star = np.clip(w_star, 0.0, None)
    w_star = w_star / w_star.sum()

    exp_ret = float(w_star @ mu_aug)
    exp_vol = float(np.sqrt(w_star @ cov_aug @ w_star))
    sharpe  = (exp_ret - risk_free_rate) / exp_vol if exp_vol > 1e-9 else 0.0

    # Realized portfolio daily returns (cash leg contributes its deterministic
    # rf/252 daily yield) — used for downside-only Sortino and historical VaR.
    risky_w     = w_star[:n_risky]
    cash_w      = float(w_star[-1])
    port_daily  = returns_df.values @ risky_w + cash_w * (risk_free_rate / TRADING_DAYS)
    neg         = port_daily[port_daily < 0]
    downside_ann = float(neg.std() * np.sqrt(TRADING_DAYS)) if len(neg) > 1 else 0.0
    sortino     = (exp_ret - risk_free_rate) / downside_ann if downside_ann > 1e-9 else 0.0
    # 95% historical 1-day VaR scaled to annual via sqrt-of-time. Stored as a
    # positive magnitude (e.g. 0.18 = "5% chance of losing >18% over a year").
    var_95      = float(-np.percentile(port_daily, 5) * np.sqrt(TRADING_DAYS))

    allocations: dict[str, dict[str, float]] = {}
    for i, t in enumerate(used):
        w = float(w_star[i])
        if w < 1e-6:
            continue
        try:
            price = float(get_stock_info(t).get("price") or 0.0)
        except Exception as e:
            log.warning("price fetch failed for %s: %s", t, e)
            price = 0.0
        dollars = w * budget
        shares = (dollars / price) if price > 0 else 0.0
        allocations[t] = {"weight": w, "dollars": dollars,
                          "shares": shares, "price": price}
    cash_dollars = float(w_star[-1]) * budget

    frontier_points = _build_frontier(mu_aug, cov_aug, max_weight, n_risky,
                                      frontier_samples, risk_free_rate)
    frontier_line   = _build_frontier_line(mu_aug, cov_aug, max_weight, n_risky)

    return {
        "allocations":     allocations,
        "cash_dollars":    cash_dollars,
        "metrics": {
            "expected_return": exp_ret,
            "expected_vol":    exp_vol,
            "sharpe":          sharpe,
            "sortino":         sortino,
            "var_95":          var_95,
            "target_vol":      target_vol,
            "risk_free_rate":  risk_free_rate,
        },
        "frontier_points": frontier_points,
        "frontier_line":   frontier_line,
        "returns_df":      returns_df,
        "warnings":        warnings,
    }


def _build_frontier(mu: np.ndarray, cov: np.ndarray, max_w_risky: float,
                    n_risky: int, samples: int,
                    risk_free_rate: float = 0.0) -> list[dict[str, float]]:
    """Monte Carlo over the *risky-only* simplex — returns vol, return, and Sharpe per point.

    CASH is excluded from sampling: mixing any risky portfolio with zero-vol CASH
    collapses all combinations onto a straight line in (vol, return) space.
    """
    if n_risky < 2:
        return []

    rng = np.random.default_rng(42)
    count = min(samples, 200_000)

    mu_r  = mu[:n_risky]
    cov_r = cov[:n_risky, :n_risky]

    raw = rng.dirichlet(np.ones(n_risky), size=count)
    # No per-asset cap here — show the full risky feasible set so the bullet
    # shape appears. The frontier line and optimized star still respect max_w_risky.

    rets   = raw @ mu_r
    vols   = np.sqrt(np.einsum("bi,ij,bj->b", raw, cov_r, raw))
    sharps = np.where(vols > 1e-9, (rets - risk_free_rate) / vols, 0.0)

    log.info("frontier: %d Monte Carlo risky-only portfolios sampled", count)
    return [{"vol": float(v), "return": float(r), "sharpe": float(s)}
            for v, r, s in zip(vols, rets, sharps)]


def _build_frontier_line(mu: np.ndarray, cov: np.ndarray,
                         max_w_risky: float, n_risky: int,
                         n_pts: int = 80) -> list[dict[str, float]]:
    """Trace the efficient frontier curve for risky assets (for the red dashed line)."""
    if n_risky < 2:
        return []

    mu_r  = mu[:n_risky]
    cov_r = cov[:n_risky, :n_risky]
    bounds = [(0.0, max_w_risky)] * n_risky
    cons_eq = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.full(n_risky, 1.0 / n_risky)

    # Minimum-variance portfolio → gives the leftmost point
    res_mv = minimize(lambda w: float(w @ cov_r @ w), x0, method="SLSQP",
                      bounds=bounds, constraints=cons_eq,
                      options={"ftol": 1e-9, "maxiter": 300})
    if not res_mv.success:
        return []
    w_mv = np.clip(res_mv.x, 0.0, None); w_mv /= w_mv.sum()
    r_min = float(w_mv @ mu_r)
    r_max = max_w_risky * float(np.max(mu_r)) + (1.0 - max_w_risky) * float(np.min(mu_r))
    if r_max <= r_min + 1e-6:
        return [{"vol": float(np.sqrt(w_mv @ cov_r @ w_mv)), "return": r_min}]

    pts: list[dict[str, float]] = [
        {"vol": float(np.sqrt(w_mv @ cov_r @ w_mv)), "return": r_min}
    ]
    x0 = w_mv.copy()
    for target_ret in np.linspace(r_min, r_max * 0.98, n_pts)[1:]:
        cons = [
            {"type": "eq",   "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w, r=target_ret: float(w @ mu_r) - r},
        ]
        res = minimize(lambda w: float(w @ cov_r @ w), x0, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"ftol": 1e-9, "maxiter": 300})
        if res.success:
            w = np.clip(res.x, 0.0, None); w /= w.sum()
            pts.append({"vol": float(np.sqrt(w @ cov_r @ w)), "return": float(w @ mu_r)})
            x0 = res.x.copy()

    pts.sort(key=lambda p: p["vol"])
    return pts


def build_plots(result: dict[str, Any]) -> tuple[go.Figure, go.Figure, go.Figure]:
    allocs = result["allocations"]
    cash   = result["cash_dollars"]
    budget = cash + sum(v["dollars"] for v in allocs.values())

    labels  = list(allocs.keys()) + [CASH]
    dollars = [v["dollars"] for v in allocs.values()] + [cash]
    weights = [d / budget if budget else 0.0 for d in dollars]

    fig_pie = go.Figure(go.Pie(
        labels=labels, values=weights, hole=0.35,
        textinfo="label+percent", textposition="inside",
        showlegend=True,
    ))
    fig_pie.update_layout(title="Allocation (weights)", template="plotly_dark")

    fig_bar = go.Figure(go.Bar(x=labels, y=dollars))
    fig_bar.update_layout(
        title="Dollar allocation",
        yaxis_title="USD",
        template="plotly_dark",
        height=420,
        autosize=True,
        margin=dict(l=60, r=24, t=48, b=48),
    )

    pts    = result["frontier_points"]
    vols   = [p["vol"]              for p in pts]
    rets   = [p["return"]           for p in pts]
    sharps = [p.get("sharpe", 0.0)  for p in pts]

    fig_f = go.Figure(go.Scatter(
        x=vols, y=rets, mode="markers",
        marker=dict(
            size=4, opacity=0.7,
            color=sharps, colorscale="Viridis",
            colorbar=dict(title="Sharpe Ratio", thickness=15, len=0.8),
        ),
        name="Portfolios",
    ))

    # Red dashed efficient frontier curve
    line_pts = result.get("frontier_line", [])
    if line_pts:
        fig_f.add_trace(go.Scatter(
            x=[p["vol"]    for p in line_pts],
            y=[p["return"] for p in line_pts],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Efficient frontier",
        ))

    m = result["metrics"]
    fig_f.add_trace(go.Scatter(
        x=[m["expected_vol"]], y=[m["expected_return"]],
        mode="markers",
        marker=dict(size=14, color="#00FF94", symbol="star",
                    line=dict(color="white", width=1)),
        name="Optimized",
    ))
    fig_f.update_layout(
        title="Efficient frontier",
        xaxis_title="Volatility (σ)",
        yaxis_title="Expected return",
        template="plotly_dark",
        height=460,
        autosize=True,
        margin=dict(l=60, r=100, t=48, b=48),
        legend=dict(
            orientation="h",
            x=0.01, y=1.08,
            xanchor="left", yanchor="bottom",
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig_pie, fig_bar, fig_f


def save_allocation(portfolio_id: int, result: dict[str, Any],
                    budget: float, target_vol: float, lookback: str,
                    commentary: str = "") -> None:
    """Persist (overwrite) the single allocation row for this portfolio."""
    from core.database import SessionLocal
    from core.models import PortfolioAllocationDB
    m = result["metrics"]
    payload = json.dumps(result["allocations"])
    frontier_payload = json.dumps(result.get("frontier_points", []))
    with SessionLocal() as s:
        row = s.get(PortfolioAllocationDB, portfolio_id)
        if row is None:
            row = PortfolioAllocationDB(portfolio_id=portfolio_id)
            s.add(row)
        row.budget           = float(budget)
        row.target_vol       = float(target_vol)
        row.lookback         = lookback
        row.expected_return  = float(m["expected_return"])
        row.expected_vol     = float(m["expected_vol"])
        row.sharpe           = float(m["sharpe"])
        row.sortino          = float(m.get("sortino", 0.0) or 0.0)
        row.var_95           = float(m.get("var_95", 0.0) or 0.0)
        row.risk_free_rate   = float(m["risk_free_rate"])
        row.cash_dollars     = float(result["cash_dollars"])
        row.allocations_json = payload
        row.frontier_json    = frontier_payload
        row.commentary       = commentary
        s.commit()
