"""Dashboard helpers: live-watchlist rows + last-plan rendering."""
from __future__ import annotations
import json
import numpy as np
import plotly.graph_objects as go


def _stock_ratios(returns: np.ndarray) -> tuple[float, float]:
    """Annual Sharpe and Sortino (risk-free=0) from a daily-returns array."""
    if len(returns) < 2:
        return 0.0, 0.0
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    neg     = returns[returns < 0]
    sortino = ann_ret / (neg.std() * np.sqrt(252)) if len(neg) > 1 else 0.0
    return round(sharpe, 2), round(sortino, 2)


def live_watchlist_rows(portfolio_id: int) -> list[list]:
    """Columns: Ticker | Price | 1d % | 1mo % | 3mo % | 1y % | Sharpe | Sortino."""
    from core.database import SessionLocal
    from core.models import HoldingDB
    from services.stock_service import get_stock_info, get_period_changes, get_historical

    with SessionLocal() as s:
        tickers = sorted(set(
            h.ticker for h in
            s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()
        ))

    # Fetch 1-year returns for each ticker (cached, no extra network cost)
    returns_map: dict[str, np.ndarray] = {}
    for t in tickers:
        hist = get_historical(t, period="1y")
        if hist is not None and not hist.empty:
            returns_map[t] = hist["Close"].pct_change().dropna().values

    stock_rows: list[list] = []
    for t in tickers:
        info    = get_stock_info(t) or {}
        periods = get_period_changes(t) or {}
        price   = float(info.get("price") or 0.0)
        sharpe, sortino = _stock_ratios(returns_map.get(t, np.array([])))
        stock_rows.append([
            t, f"${price:.2f}",
            f"{periods.get('change_1d_pct',  0.0):+.2f}%",
            f"{periods.get('change_1mo_pct', 0.0):+.2f}%",
            f"{periods.get('change_3mo_pct', 0.0):+.2f}%",
            f"{periods.get('change_1y_pct',  0.0):+.2f}%",
            f"{sharpe:.2f}", f"{sortino:.2f}",
        ])

    # Equal-weighted portfolio row
    if returns_map:
        min_len  = min(len(v) for v in returns_map.values())
        port_r   = np.mean([v[-min_len:] for v in returns_map.values()], axis=0)
        p_sharpe, p_sortino = _stock_ratios(port_r)
        eq_row = ["Portfolio (eq-wt)", "—", "—", "—", "—", "—",
                  f"{p_sharpe:.2f}", f"{p_sortino:.2f}"]
    else:
        eq_row = ["Portfolio (eq-wt)", "—", "—", "—", "—", "—", "—", "—"]

    # Optimized-weighted portfolio row (only when a saved plan exists)
    from core.database import SessionLocal as _SL
    from core.models import PortfolioAllocationDB as _ADB
    with _SL() as s:
        alloc_row = s.get(_ADB, portfolio_id)

    if alloc_row and returns_map:
        allocs  = json.loads(alloc_row.allocations_json)
        total_w = sum(v["weight"] for v in allocs.values())
        if total_w > 0:
            common = [t for t in allocs if t in returns_map]
            if common:
                min_len = min(len(returns_map[t]) for t in common)
                w       = np.array([allocs[t]["weight"] / total_w for t in common])
                w      /= w.sum()
                stacked = np.column_stack([returns_map[t][-min_len:] for t in common])
                opt_r   = stacked @ w
                _, o_sortino = _stock_ratios(opt_r)
                # Use the Sharpe saved by the optimizer (same formula/inputs as the
                # "Last optimized plan" metric box) so both displays agree.
                o_sharpe = alloc_row.sharpe
                opt_row = ["Portfolio (optimized)", "—", "—", "—", "—", "—",
                           f"{o_sharpe:.2f}", f"{o_sortino:.2f}"]
            else:
                opt_row = None
        else:
            opt_row = None
    else:
        opt_row = None

    rows = (
        [["CASH", "$1.00", "+0.00%", "+0.00%", "+0.00%", "+0.00%", "—", "—"]]
        + stock_rows
        + [eq_row]
    )
    if opt_row:
        rows.append(opt_row)
    return rows


def last_plan_rows(portfolio_id: int) -> tuple[list[list], dict | None]:
    """Returns (dollar_rows, metrics) — rows include CASH; metrics None if no plan."""
    from core.database import SessionLocal
    from core.models import PortfolioAllocationDB
    with SessionLocal() as s:
        row = s.get(PortfolioAllocationDB, portfolio_id)
        if row is None:
            return [], None
        allocs = json.loads(row.allocations_json)
        rows = []
        for ticker, v in allocs.items():
            rows.append([ticker,
                         f"{v['weight']*100:.2f}%",
                         f"${v['dollars']:,.0f}",
                         f"{v['shares']:.2f}",
                         f"${v['price']:.2f}"])
        rows.insert(0, ["CASH", f"{(row.cash_dollars/row.budget)*100:.2f}%",
                        f"${row.cash_dollars:,.0f}", "—", "$1.00"])
        metrics = {
            "budget":          row.budget,
            "expected_return": row.expected_return,
            "expected_vol":    row.expected_vol,
            "sharpe":          row.sharpe,
            "cash_dollars":    row.cash_dollars,
            "created_at":      row.created_at,
        }
        return rows, metrics


def last_plan_pie(portfolio_id: int) -> go.Figure | None:
    rows, metrics = last_plan_rows(portfolio_id)
    if not rows:
        return None
    labels = [r[0] for r in rows]
    values = [float(r[1].rstrip("%")) for r in rows]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.35,
        textinfo="label+percent", textposition="inside",
        showlegend=True,
    ))
    fig.update_layout(
        title="Last optimized allocation",
        template="plotly_dark",
        height=420,
        autosize=True,
        margin=dict(l=24, r=24, t=48, b=24),
    )
    return fig
