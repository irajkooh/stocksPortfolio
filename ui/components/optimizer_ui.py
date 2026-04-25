"""Optimizer tab helpers: rf slider/textbox sync + Optimize button handler."""
from __future__ import annotations

import gradio as gr

from services.optimizer import optimize_portfolio, build_plots, save_allocation
from services.parsing import parse_rf


def sync_slider_to_text(pct: float) -> str:
    return f"{float(pct):.2f}%"


def sync_text_to_slider(text: str, current_pct: float):
    try:
        dec = parse_rf(text)
    except Exception:
        gr.Warning(f"Invalid rate: {text!r}")
        return gr.update(), gr.update(value=f"{current_pct:.2f}%")
    return gr.update(value=dec * 100), gr.update(value=f"{dec * 100:.2f}%")


_DASH_EMPTY = ("—", "—", "—", "—", "—", None, [], "_Run the Optimizer to see your plan._", None)


def run_optimize(budget, target_vol_pct, rf_text, lookback, frontier_samples,
                 portfolio_id):
    from core.database import SessionLocal
    from core.models import HoldingDB
    from ui.components.dashboard import live_watchlist_rows, last_plan_rows, last_plan_pie

    def _dash_outputs():
        from ui.frontend import _portfolio_vs_spy_fig
        watch = live_watchlist_rows(portfolio_id)
        rows, m = last_plan_rows(portfolio_id)
        if m is None:
            return (watch,) + _DASH_EMPTY
        vs_spy = _portfolio_vs_spy_fig(portfolio_id)
        return (
            watch,
            f"${m['budget']:,.0f}",
            f"{m['expected_return']*100:.2f}%",
            f"{m['expected_vol']*100:.2f}%",
            f"{m['sharpe']:.3f}",
            f"${m['cash_dollars']:,.0f}",
            last_plan_pie(portfolio_id),
            rows,
            f"_Last optimized: {m['created_at'].strftime('%Y-%m-%d %H:%M:%S')}_",
            vs_spy,
        )

    with SessionLocal() as s:
        tickers = [
            h.ticker
            for h in s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()
        ]

    try:
        rf = parse_rf(rf_text)
    except Exception as e:
        return (f"❌ {e}", "", "", "", "", None, None, None, "") + (None,) + _DASH_EMPTY

    try:
        result = optimize_portfolio(
            tickers=tickers,
            budget=float(budget),
            target_vol=float(target_vol_pct) / 100.0,
            lookback=lookback,
            risk_free_rate=rf,
            frontier_samples=int(frontier_samples),
        )
    except Exception as e:
        return (f"❌ {e}", "", "", "", "", None, None, None, "") + (None,) + _DASH_EMPTY

    fig_p, fig_b, fig_f = build_plots(result)
    save_allocation(
        portfolio_id,
        result,
        budget=float(budget),
        target_vol=float(target_vol_pct) / 100.0,
        lookback=lookback,
    )

    m = result["metrics"]
    commentary = "\n\n".join([f"- {w}" for w in result["warnings"]]) or \
                 "Optimization complete."

    return (
        "✅ Optimized",
        f"{m['expected_return']*100:.2f}%",
        f"{m['expected_vol']*100:.2f}%",
        f"{m['sharpe']:.3f}",
        f"${result['cash_dollars']:,.0f}",
        fig_p, fig_b, fig_f,
        commentary,
    ) + _dash_outputs()
