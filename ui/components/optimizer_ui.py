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


_DASH_EMPTY = (
    "—", "—", "—",
    gr.update(value="—", label="Sharpe (ann.)"),
    gr.update(value="—", label="Sortino (ann.)"),
    "—", "—", None, "", "_Run the Optimizer to see your plan._", None,
    gr.update(), gr.update(),  # watch_label, spy_label
)


def frontier_click(click_data: str):
    """JS bridge: click_data is 'vol,ret,timestamp' written by plotly_click handler."""
    _hide = (0.0, gr.update(visible=False), gr.update(visible=False), "")
    if not click_data or "," not in click_data:
        return _hide
    try:
        parts = click_data.split(",")
        vol = float(parts[0])
        ret = float(parts[1]) if len(parts) > 1 else 0.0
        if vol <= 0:
            return _hide
        label = (
            f"**Optimize at this point?**  "
            f"Target vol: **{vol*100:.2f}%** · "
            f"Expected return: **{ret*100:.2f}%**"
        )
        return vol, gr.update(visible=True), gr.update(visible=True), label
    except Exception:
        return _hide


_last_frontier_ts: dict[str, str] = {"v": ""}


def frontier_confirm(click_data, budget, rf_text, lookback, frontier_samples,
                     portfolio_id, sr_threshold=1.0):
    """Single-shot: parse vol from frontier click_data and run full optimization."""
    # Dedup: both .change() (Gradio 6.8/local) and .click() (Gradio 6.9/HF Space)
    # fire for every point click — only run once per unique timestamp.
    try:
        ts = str(click_data).split(",")[2] if click_data and "," in str(click_data) else ""
    except Exception:
        ts = ""
    if ts and ts == _last_frontier_ts["v"]:
        print(f"[frontier_confirm] duplicate ts={ts!r} — skipping")
        return tuple(gr.update() for _ in range(26))
    _last_frontier_ts["v"] = ts

    print(f"[frontier_confirm] called: click_data={click_data!r} budget={budget} portfolio_id={portfolio_id}")
    _noop = tuple(gr.update() for _ in range(26))
    if not click_data or "," not in str(click_data):
        print("[frontier_confirm] returning _noop: bad click_data")
        return _noop
    try:
        vol = float(str(click_data).split(",")[0])
    except Exception as e:
        print(f"[frontier_confirm] returning _noop: parse error {e}")
        return _noop
    if vol <= 0:
        print(f"[frontier_confirm] returning _noop: vol={vol} <= 0")
        return _noop
    print(f"[frontier_confirm] calling run_optimize with vol={vol*100:.2f}%")
    result = run_optimize(budget, round(vol * 100.0, 2), rf_text, lookback,
                          frontier_samples, portfolio_id, sr_threshold,
                          force_target_vol=True)
    print(f"[frontier_confirm] run_optimize returned {len(result)} items, first={result[0]!r}")
    return result


def run_optimize(budget, target_vol_pct, rf_text, lookback, frontier_samples,
                 portfolio_id, sr_threshold=1.0, force_target_vol=False):
    from core.database import SessionLocal
    from core.models import HoldingDB
    from ui.components.dashboard import live_watchlist_rows, last_plan_rows, last_plan_pie

    def _dash_outputs(saved_at):
        from ui.frontend import _portfolio_vs_spy_fig, _watchlist_html, _watch_headers, _ALLOC_HEADERS
        import pandas as pd
        watch = [r for r in live_watchlist_rows(portfolio_id) if not (len(r) > 8 and r[8] == "red")]
        rows, m = last_plan_rows(portfolio_id)
        if m is None:
            return (_watchlist_html(watch, _watch_headers()),) + _DASH_EMPTY
        stored_at    = m["created_at"] or saved_at
        opt_date_str = stored_at.strftime("%Y-%m-%d")
        vs_spy = _portfolio_vs_spy_fig(portfolio_id, opt_date_override=pd.Timestamp(stored_at.date()))
        sortino_s = f"{m['sortino']:.3f}" if m.get("sortino") is not None else "—"
        var_s     = f"{m['var_95']*100:.2f}%" if m.get("var_95") is not None else "—"
        return (
            _watchlist_html(watch, _watch_headers()),
            f"${m['budget']:,.0f}",
            f"{m['expected_return']*100:.2f}%",
            f"{m['expected_vol']*100:.2f}%",
            gr.update(value=f"{m['sharpe']:.3f}", label=f"Sharpe (ann.)\nat {opt_date_str}"),
            gr.update(value=sortino_s, label=f"Sortino (ann.)\nat {opt_date_str}"),
            var_s,
            f"${m['cash_dollars']:,.0f}",
            last_plan_pie(portfolio_id),
            _watchlist_html(rows, _ALLOC_HEADERS),
            f"_Last optimized: {stored_at.strftime('%Y-%m-%d %H:%M:%S')}_",
            vs_spy,
            gr.update(),  # watch_label — preserve existing value
            gr.update(),  # spy_label — preserve existing value
        )

    with SessionLocal() as s:
        tickers = [
            h.ticker
            for h in s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()
        ]

    try:
        rf = parse_rf(rf_text)
    except Exception as e:
        out = (f"❌ {e}", "", "", gr.update(value="", label="Sharpe (ann.)"), gr.update(value="", label="Sortino (ann.)"), "", "", None, None, None, "", gr.update()) + ("",) + _DASH_EMPTY
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
            force_target_vol=force_target_vol,
        )
    except Exception as e:
        out = (f"❌ {e}", "", "", gr.update(value="", label="Sharpe (ann.)"), gr.update(value="", label="Sortino (ann.)"), "", "", None, None, None, "", gr.update()) + ("",) + _DASH_EMPTY
        result = out + ("",) * (26 - len(out))
        return result[:26]

    from datetime import datetime as _dt
    fig_p, fig_b, fig_f = build_plots(result_obj)
    saved_at = _dt.utcnow()

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

    # Auto-update slider to match actual portfolio vol when it exceeds target;
    # save the raised value so it's restored on next portfolio load.
    new_slider_val = max(actual_vol_pct, target_vol_pct2)

    save_allocation(
        portfolio_id,
        result_obj,
        budget=float(budget),
        target_vol=new_slider_val / 100.0,   # save the raised vol, not the raw input
        lookback=lookback,
        frontier_samples=int(frontier_samples),
        sr_threshold=float(sr_threshold),
        commentary=commentary,               # persist warning message
    )

    sharpe_s  = f"{m['sharpe']:.3f}"
    sortino_s = f"{m['sortino']:.3f}" if m.get("sortino") is not None else "—"
    var_s = f"{m['var_95']*100:.2f}%" if m.get("var_95") is not None else "—"
    opt_date_str = saved_at.strftime("%Y-%m-%d")
    out = (
        "✅ Optimized",
        f"{m['expected_return']*100:.2f}%",
        f"{m['expected_vol']*100:.2f}%",
        gr.update(value=sharpe_s,  label=f"Sharpe (ann.)\nat {opt_date_str}"),
        gr.update(value=sortino_s, label=f"Sortino (ann.)\nat {opt_date_str}"),
        var_s,
        f"${result_obj['cash_dollars']:,.0f}",
        fig_p, fig_b, fig_f,
        commentary,
        gr.update(value=new_slider_val),   # opt_target_vol slider
    ) + _dash_outputs(saved_at)
    # Final safeguard: always return exactly 26 outputs
    result = out + ("",) * (26 - len(out))
    return result[:26]
