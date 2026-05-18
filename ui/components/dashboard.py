"""Dashboard helpers: live-watchlist rows + last-plan rendering."""
from __future__ import annotations
import json
import numpy as np
import plotly.graph_objects as go


def _stock_ratios(returns: np.ndarray, rf: float = 0.0) -> tuple[float, float]:
    """Annual Sharpe and Sortino from a daily-returns array."""
    if len(returns) < 2:
        return 0.0, 0.0
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    excess  = ann_ret - rf
    sharpe  = excess / ann_vol if ann_vol > 0 else 0.0
    neg     = returns[returns < 0]
    sortino = excess / (neg.std() * np.sqrt(252)) if len(neg) > 1 else 0.0
    return round(sharpe, 2), round(sortino, 2)


def live_watchlist_rows(portfolio_id: int) -> list[list]:
    """Columns: Ticker | Price | 1d % | 1mo % | 3mo % | 1y % | Sharpe | Sortino."""
    from core.database import SessionLocal
    from core.models import HoldingDB
    from services.stock_service import get_stock_info, get_period_changes, get_historical

    from core.models import PortfolioAllocationDB as _ADB
    with SessionLocal() as s:
        all_holdings = s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()
        alloc_row    = s.get(_ADB, portfolio_id)

    # Always show all tickers; track which ones the optimizer selected
    opt_tickers: set[str] = set()
    if alloc_row:
        opt_tickers = set(json.loads(alloc_row.allocations_json).keys())
    bought  = [h for h in all_holdings if h.shares is not None and h.shares > 0]
    tickers = sorted(h.ticker for h in (bought if bought else all_holdings))

    rf = alloc_row.risk_free_rate if alloc_row else 0.04

    # Fetch histories and period returns for each ticker in one pass
    returns_map: dict[str, np.ndarray] = {}
    period_map: dict[str, dict[str, float]] = {}
    for t in tickers:
        hist = get_historical(t, period="1y")
        if hist is not None and not hist.empty:
            returns_map[t] = hist["Close"].pct_change().dropna().values
        p = get_period_changes(t) or {}
        period_map[t] = {
            "1d":  float(p.get("change_1d_pct",  0.0) or 0.0),
            "1mo": float(p.get("change_1mo_pct", 0.0) or 0.0),
            "3mo": float(p.get("change_3mo_pct", 0.0) or 0.0),
            "1y":  float(p.get("change_1y_pct",  0.0) or 0.0),
        }

    stock_rows: list[list] = []
    for t in tickers:
        info    = get_stock_info(t) or {}
        price   = float(info.get("price") or 0.0)
        sharpe, sortino = _stock_ratios(returns_map.get(t, np.array([])), rf)
        pm = period_map[t]
        color_hint = "green" if (opt_tickers and t in opt_tickers) else ("red" if opt_tickers else "")
        stock_rows.append([
            t, f"${price:.2f}",
            f"{pm['1d']:+.2f}%",
            f"{pm['1mo']:+.2f}%",
            f"{pm['3mo']:+.2f}%",
            f"{pm['1y']:+.2f}%",
            f"{sharpe:.2f}", f"{sortino:.2f}",
            color_hint,
        ])

    # Optimizer winners (green) first sorted by Sharpe desc, reds at bottom also by Sharpe desc
    if opt_tickers:
        def _row_sort_key(r):
            try:
                sharpe = float(r[6])
            except (ValueError, TypeError):
                sharpe = 0.0
            return (0 if r[8] == "green" else 1, -sharpe)
        stock_rows.sort(key=_row_sort_key)

    # Equal-weighted portfolio row
    if tickers and returns_map:
        min_len  = min(len(v) for v in returns_map.values())
        port_r   = np.mean([v[-min_len:] for v in returns_map.values()], axis=0)
        p_sharpe, p_sortino = _stock_ratios(port_r, rf)
        n = len(tickers)
        eq_row = [
            "Portfolio (eq-wt)", "—",
            f"{sum(period_map[t]['1d']  for t in tickers)/n:+.2f}%",
            f"{sum(period_map[t]['1mo'] for t in tickers)/n:+.2f}%",
            f"{sum(period_map[t]['3mo'] for t in tickers)/n:+.2f}%",
            f"{sum(period_map[t]['1y']  for t in tickers)/n:+.2f}%",
            f"{p_sharpe:.2f}", f"{p_sortino:.2f}",
        ]
    else:
        eq_row = ["Portfolio (eq-wt)", "—", "—", "—", "—", "—", "—", "—"]
    cash_1d  = rf / 252
    cash_1mo = rf / 12
    cash_3mo = rf / 4
    cash_1y  = rf

    if alloc_row:
        allocs    = json.loads(alloc_row.allocations_json)
        cash_w    = alloc_row.cash_dollars / alloc_row.budget if alloc_row.budget else 0.0
        opt_1d = opt_1mo = opt_3mo = opt_1y = 0.0
        for ticker, v in allocs.items():
            w  = float(v["weight"])
            pm = period_map.get(ticker)
            if pm:
                opt_1d  += w * pm["1d"]
                opt_1mo += w * pm["1mo"]
                opt_3mo += w * pm["3mo"]
                opt_1y  += w * pm["1y"]
        # Add cash contribution (annualised rf scaled to each period)
        opt_1d  += cash_w * cash_1d  * 100
        opt_1mo += cash_w * cash_1mo * 100
        opt_3mo += cash_w * cash_3mo * 100
        opt_1y  += cash_w * cash_1y  * 100

        o_sharpe  = round(float(alloc_row.sharpe  or 0.0), 2)
        o_sortino = round(float(alloc_row.sortino or 0.0), 2)

        opt_row = [
            "Portfolio (optimized)", "—",
            f"{opt_1d:+.2f}%", f"{opt_1mo:+.2f}%", f"{opt_3mo:+.2f}%", f"{opt_1y:+.2f}%",
            f"{o_sharpe:.2f}", f"{o_sortino:.2f}",
        ]
    else:
        opt_row = None

    rows = (
        [["CASH", "$1.00",
          f"+{cash_1d*100:.3f}%",
          f"+{cash_1mo*100:.2f}%",
          f"+{cash_3mo*100:.2f}%",
          f"+{cash_1y*100:.2f}%",
          "0.00", "0.00"]]
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
        from services.stock_service import get_historical
        rf = row.risk_free_rate if row.risk_free_rate else 0.04
        sharpe_map: dict[str, float] = {}
        for ticker in allocs:
            hist = get_historical(ticker, period="1y")
            if hist is not None and not hist.empty:
                rets = hist["Close"].pct_change().dropna().values
                sharpe_map[ticker], _ = _stock_ratios(rets, rf)

        port_sharpe  = round(float(row.sharpe  or 0.0), 2)
        port_sortino = round(float(row.sortino or 0.0), 2)

        rows = []
        for ticker, v in allocs.items():
            rows.append([ticker,
                         f"{v['weight']*100:.2f}%",
                         f"${v['dollars']:,.0f}",
                         f"{v['shares']:.2f}",
                         f"${v['price']:.2f}"])
        rows.sort(key=lambda r: sharpe_map.get(r[0], 0.0), reverse=True)
        rows.insert(0, ["CASH", f"{(row.cash_dollars/row.budget)*100:.2f}%",
                        f"${row.cash_dollars:,.0f}", "—", "$1.00"])
        from datetime import datetime as _dt
        _utc_off = _dt.now() - _dt.utcnow()
        metrics = {
            "budget":          row.budget,
            "expected_return": row.expected_return,
            "expected_vol":    row.expected_vol,
            "sharpe":          port_sharpe,
            "sortino":         port_sortino,
            "var_95":          row.var_95,
            "cash_dollars":    row.cash_dollars,
            "created_at":      (row.created_at + _utc_off) if row.created_at else None,
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
