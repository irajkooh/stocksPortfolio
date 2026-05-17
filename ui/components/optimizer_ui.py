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


def sync_sr_slider_to_text(val: float) -> str:
    return f"{float(val):.2f}"


def sync_sr_text_to_slider(text: str, current: float):
    try:
        val = float(str(text).replace("%", "").strip())
        val = max(0.0, val)
    except Exception:
        gr.Warning(f"Invalid SR threshold: {text!r}")
        return gr.update(), gr.update(value=f"{current:.2f}")
    return gr.update(value=val), gr.update(value=f"{val:.2f}")


_DASH_EMPTY = ("—", "—", "—", "—", "—", "—", "—", None, "", "_Run the Optimizer to see your plan._", None)


def run_optimize(budget, target_vol_pct, rf_text, lookback, frontier_samples,
                 portfolio_id, sr_threshold=1.0):
    from core.database import SessionLocal
    from core.models import HoldingDB
    from ui.components.dashboard import live_watchlist_rows, last_plan_rows, last_plan_pie

    def _dash_outputs(saved_at):
        from ui.frontend import _portfolio_vs_spy_fig, _watchlist_html, _DASH_WATCH_HEADERS, _ALLOC_HEADERS
        import pandas as pd
        watch = [r for r in live_watchlist_rows(portfolio_id) if not (len(r) > 8 and r[8] == "red")]
        rows, m = last_plan_rows(portfolio_id)
        if m is None:
            return (_watchlist_html(watch, _DASH_WATCH_HEADERS),) + _DASH_EMPTY
        vs_spy = _portfolio_vs_spy_fig(portfolio_id, opt_date_override=pd.Timestamp(saved_at.date()))
        sortino_s = f"{m['sortino']:.3f}" if m.get("sortino") is not None else "—"
        var_s     = f"{m['var_95']*100:.2f}%" if m.get("var_95") is not None else "—"
        return (
            _watchlist_html(watch, _DASH_WATCH_HEADERS),
            f"${m['budget']:,.0f}",
            f"{m['expected_return']*100:.2f}%",
            f"{m['expected_vol']*100:.2f}%",
            f"{m['sharpe']:.3f}",
            sortino_s,
            var_s,
            f"${m['cash_dollars']:,.0f}",
            last_plan_pie(portfolio_id),
            _watchlist_html(rows, _ALLOC_HEADERS),
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
        out = (f"❌ {e}", "", "", "", "", "", "", None, None, None, "", gr.update()) + ("",) + _DASH_EMPTY
        result = out + ("",) * (26 - len(out))
        return result[:26]

    try:
        result_obj = optimize_portfolio(
            tickers=tickers,
            budget=float(budget),
            target_vol=float(target_vol_pct) / 100.0,
            lookback=lookback,
            risk_free_rate=rf,
            frontier_samples=int(frontier_samples),
            sharpe_hurdle=float(sr_threshold),
        )
    except Exception as e:
        out = (f"❌ {e}", "", "", "", "", "", "", None, None, None, "", gr.update()) + ("",) + _DASH_EMPTY
        result = out + ("",) * (26 - len(out))
        return result[:26]

    from datetime import datetime as _dt
    fig_p, fig_b, fig_f = build_plots(result_obj)
    saved_at = _dt.utcnow()
    save_allocation(
        portfolio_id,
        result_obj,
        budget=float(budget),
        target_vol=float(target_vol_pct) / 100.0,
        lookback=lookback,
    )

    m = result_obj["metrics"]
    actual_vol_pct  = m["expected_vol"] * 100.0
    target_vol_pct2 = float(target_vol_pct)

    # Build commentary — red warning when actual vol exceeds the target setting
    warn_lines = [f"- {w}" for w in result_obj["warnings"]]
    if actual_vol_pct > target_vol_pct2 + 0.5:
        vol_warn = (
            f'<span style="color:#FF4444;font-weight:bold;">'
            f'⚠️ Target Risk raised from {target_vol_pct2:.1f}% → {actual_vol_pct:.1f}%: '
            f'the max-Sharpe portfolio natural volatility is {actual_vol_pct:.1f}%. '
            f'No cash is held because all stocks cleared the Sharpe hurdle — '
            f'raising Target Risk to ≥ {actual_vol_pct:.1f}% reflects the true portfolio.'
            f'</span>'
        )
        warn_lines.insert(0, vol_warn)
    commentary = "\n\n".join(warn_lines) or "Optimization complete."

    # Auto-update slider to match actual portfolio vol when it exceeds target
    new_slider_val = max(actual_vol_pct, target_vol_pct2)

    sortino_s = f"{m['sortino']:.3f}" if m.get("sortino") is not None else "—"
    var_s     = f"{m['var_95']*100:.2f}%" if m.get("var_95") is not None else "—"
    out = (
        "✅ Optimized",
        f"{m['expected_return']*100:.2f}%",
        f"{m['expected_vol']*100:.2f}%",
        f"{m['sharpe']:.3f}",
        sortino_s,
        var_s,
        f"${result_obj['cash_dollars']:,.0f}",
        fig_p, fig_b, fig_f,
        commentary,
        gr.update(value=new_slider_val),   # opt_target_vol slider
    ) + _dash_outputs(saved_at)
    # Final safeguard: always return exactly 26 outputs
    result = out + ("",) * (26 - len(out))
    return result[:26]
