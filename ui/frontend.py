"""Main Gradio Blocks layout — multi-agent powered, multi-portfolio."""
import logging
import os
from datetime import date, timedelta
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import ValidationError

from core.database import SessionLocal, init_db
from core.models import HoldingCreate, HoldingDB, PortfolioDB
from services.llm_service import llm_display_name
from ui.components.dashboard import (
    live_watchlist_rows,
    last_plan_rows,
    last_plan_pie,
)
from ui.components.chatbot import run_agents, tts_html, tts_text_for_js, agent_badges_html
from ui.components.optimizer_ui import (
    run_optimize,
    frontier_confirm,
    sync_slider_to_text,
    sync_text_to_slider,
    sync_sr_slider_to_text,
    sync_sr_text_to_slider,
)
from core import runtime
from core import config as _cfg

log = logging.getLogger(__name__)

# ── Mermaid agent-workflow diagram ────────────────────────────────────────────
_MERMAID_HTML = """
<div class="mermaid-wrap">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true,theme:'dark',
  themeVariables:{primaryColor:'#00D4FF',primaryTextColor:'#fff',
  primaryBorderColor:'#374151',lineColor:'#7B2FBE',
  secondaryColor:'#111827',tertiaryColor:'#1F2937'}});</script>
<div class="mermaid">
flowchart TD
    U([👤 User Message]) --> S

    S["🎯 Supervisor Agent\n(LangGraph Router)\nReads intent · picks agents"]

    S -->|market prices / tickers| MI["📈 Market Intel Agent\nyfinance · live prices\ncompany info"]
    S -->|portfolio health| PA["💼 Portfolio Analyst\nSQLite · P&L · allocation\nposition sizing"]
    S -->|risk / volatility| RM["🛡️ Risk Manager\nSharpe · Sortino · VaR\nMax Drawdown · Beta"]
    S -->|rebalance / optimise| RL["📐 Optimizer Agent\nMarkowitz · scipy SLSQP\nBudget → $ allocations\nAUTO-TRIGGERED"]
    S -->|finance concepts| KB["📚 Knowledge Base\nChromaDB · RAG\nWikipedia · Groq/Ollama"]

    MI & PA & RM & RL & KB --> SY

    SY["✨ Synthesizer Agent\nMerges all outputs\nFormats final response"]
    SY --> R([💬 Response + Charts])

    style S  fill:#1F2937,stroke:#00D4FF,color:#fff
    style MI fill:#111827,stroke:#374151,color:#9CA3AF
    style PA fill:#111827,stroke:#374151,color:#9CA3AF
    style RM fill:#111827,stroke:#374151,color:#9CA3AF
    style RL fill:#1a1230,stroke:#7B2FBE,color:#A78BFA,font-weight:bold
    style KB fill:#111827,stroke:#374151,color:#9CA3AF
    style SY fill:#0d1929,stroke:#00FF94,color:#00FF94
    style U  fill:#0A0E1A,stroke:#374151,color:#fff
    style R  fill:#0A0E1A,stroke:#374151,color:#fff
</div>
</div>
"""


def _llm_label() -> str:
    if _cfg.GROQ_API_KEY:
        return _cfg.GROQ_MODEL
    return f"ollama:{_cfg.OLLAMA_MODEL}"


def _env_label() -> str:
    if _cfg.IS_HF_SPACE:
        return f"HF Space: {os.environ.get('SPACE_ID', '<unknown>')}"
    return "locally"


def _runtime_banner_html() -> str:
    text = (
        f"Running: {_env_label()} | "
        f"Device: {runtime.DEVICE} | "
        f"LLM: {_llm_label()}"
    )
    return f'<div class="runtime-banner">{text}</div>'

_EMPTY = go.Figure()
_EMPTY.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
                     font=dict(color="white"))


def _placeholder(height: int) -> go.Figure:
    """Blank dark figure with a fixed height — used to pre-size gr.Plot containers."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
        font=dict(color="white"),
        height=height,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def _date_range_label(heading: str, period_days: int = 365) -> str:
    today = date.today()
    start = today - timedelta(days=period_days)
    return (
        f"### {heading} "
        f"<small style='color:#6B7280;font-weight:normal;'>"
        f"({start.strftime('%b %d, %Y')} — {today.strftime('%b %d, %Y')})"
        f"</small>"
    )


# ── Portfolio management helpers ──────────────────────────────────────────────

def _list_portfolios() -> list:
    db = SessionLocal()
    try:
        return db.query(PortfolioDB).order_by(PortfolioDB.name).all()
    finally:
        db.close()


def _portfolio_choices() -> list[str]:
    return [p.name for p in _list_portfolios()]


def _id_from_choice(choice: str | None) -> int:
    if not choice:
        return 1
    with SessionLocal() as db:
        p = db.query(PortfolioDB).filter(PortfolioDB.name == choice).first()
        return p.id if p else 1


def create_portfolio(name: str):
    name = (name or "").strip().upper()
    if not name:
        return gr.update(), gr.update(), "❌ Enter a portfolio name."
    db = SessionLocal()
    try:
        if db.query(PortfolioDB).filter(PortfolioDB.name == name).first():
            return gr.update(), gr.update(), f"❌ **{name}** already exists."
        p = PortfolioDB(name=name)
        db.add(p)
        db.commit()
        db.refresh(p)
        choices = _portfolio_choices()
        return (
            gr.update(choices=choices, value=p.name),
            p.id,
            f"✅ Created **{name}**.",
        )
    except Exception as e:
        return gr.update(), gr.update(), f"❌ {e}"
    finally:
        db.close()


def delete_portfolio(portfolio_id: int):
    db = SessionLocal()
    try:
        p = db.query(PortfolioDB).filter(PortfolioDB.id == portfolio_id).first()
        if not p:
            return gr.update(), portfolio_id, "❌ Portfolio not found."
        name = p.name
        db.delete(p)
        db.commit()

        choices = _portfolio_choices()
        if not choices:
            new_p = PortfolioDB(name="DEFAULT")
            db.add(new_p)
            db.commit()
            db.refresh(new_p)
            choices = _portfolio_choices()
            new_id = new_p.id
        else:
            new_id = _id_from_choice(choices[0])

        return (
            gr.update(choices=choices, value=choices[0]),
            new_id,
            f"✅ Deleted **{name}**.",
        )
    except Exception as e:
        return gr.update(), portfolio_id, f"❌ {e}"
    finally:
        db.close()


def rename_portfolio(portfolio_id: int, new_name: str):
    new_name = (new_name or "").strip().upper()
    if not new_name:
        return gr.update(), "❌ Enter a new name."
    db = SessionLocal()
    try:
        p = db.query(PortfolioDB).filter(PortfolioDB.id == portfolio_id).first()
        if not p:
            return gr.update(), "❌ Portfolio not found."
        p.name = new_name
        db.commit()
        db.refresh(p)
        choices = _portfolio_choices()
        return (
            gr.update(choices=choices, value=p.name),
            f"✅ Renamed to **{new_name}**.",
        )
    except Exception as e:
        return gr.update(), f"❌ {e}"
    finally:
        db.close()


# ── DB helpers ────────────────────────────────────────────────────────────────

def _watchlist_df(portfolio_id: int) -> list[list]:
    """Return rows for bought stocks; CASH pinned at top. 8-col slice + color hint."""
    full_rows = live_watchlist_rows(portfolio_id)
    result = []
    for r in full_rows:
        if str(r[0]).startswith("Portfolio"):
            continue
        hint = r[8] if len(r) > 8 else ""
        result.append(r[:8] + [hint])
    return result


def _all_tickers(portfolio_id: int) -> list[str]:
    """All tickers in the watchlist regardless of shares (for the remove dropdown)."""
    with SessionLocal() as s:
        return sorted(
            h.ticker for h in
            s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()
        )


# ── HTML table helpers (replaces gr.Dataframe for mobile scroll support) ──────

_ALLOC_HEADERS = ["Ticker", "Weight", "Dollars", "Shares", "Price"]


def _watch_headers() -> list[str]:
    """Return watchlist column headers with today's date on the live ratio columns."""
    import datetime
    today = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %Z")
    return ["Ticker", "Price", "1d %", "1mo %", "3mo %", "1y %",
            f"Sharpe (ann., 1y)<br>at {today}", f"Sortino (ann., 1y)<br>at {today}"]

_TH_BASE = ("background:#1F2937;font-size:.75rem;"
            "text-transform:uppercase;letter-spacing:.05em;"
            "padding:8px 12px;white-space:nowrap;text-align:left;border:none;")
_TD_BASE = "padding:6px 12px;white-space:nowrap;border-bottom:1px solid #1F2937;"


def _watchlist_html(rows: list[list], headers: list[str]) -> str:
    """Render rows + headers as a mobile-scrollable HTML table with inline styles."""
    n      = len(headers)
    ratios = n >= 8  # last 2 cols are Sharpe / Sortino only in the 8-col table

    # ── Header row ──────────────────────────────────────────────────────────
    ths = []
    for j, h in enumerate(headers):
        color = "#A78BFA" if (ratios and j >= n - 2) else "#9CA3AF"
        ths.append(f'<th style="{_TH_BASE}color:{color};">{h}</th>')

    # ── Body rows ────────────────────────────────────────────────────────────
    trs = []
    for i, row in enumerate(rows):
        label      = str(row[0]) if row else ""
        is_opt     = label.startswith("Portfolio (optimized)")
        is_eq      = label.startswith("Portfolio (eq-wt)")
        color_hint = str(row[n]) if len(row) > n else ""   # extra hint column

        row_bg = (
            "background:#1a1800;" if is_opt else
            "background:#0d1020;" if is_eq  else
            "background:rgba(0,180,80,.07);"  if color_hint == "green" else
            "background:rgba(220,50,50,.07);" if color_hint == "red"   else
            "background:#0d1118;" if i % 2 else ""
        )

        cells = []
        for j, cell in enumerate(row[:n]):   # only render header columns
            is_ratio = ratios and j >= n - 2

            if is_opt and is_ratio:
                color = "color:#FFD700;"
                extra = "font-weight:700;"
            elif is_ratio:
                color = "color:#A78BFA;"
                extra = ""
            elif is_opt:
                color = "color:#FFD700;"
                extra = "font-weight:700;" if j == 0 else ""
            elif is_eq:
                color = "color:#00D4FF;"
                extra = ""
            elif color_hint == "green":
                color = "color:#4ADE80;"
                extra = "font-weight:700;" if j == 0 else ""
            elif color_hint == "red":
                color = "color:#F87171;"
                extra = "font-weight:700;" if j == 0 else ""
            else:
                color = "color:#E5E7EB;"
                extra = ""

            if is_opt:
                content = f'<b><span style="color:#FFD700;">{cell}</span></b>'
            elif color_hint == "green":
                content = f'<span style="color:#4ADE80;">{cell}</span>' if j == 0 else str(cell)
            elif color_hint == "red":
                content = f'<span style="color:#F87171;">{cell}</span>' if j == 0 else str(cell)
            else:
                content = str(cell)
            cells.append(
                f'<td style="{_TD_BASE}{row_bg}{color}{extra}">{content}</td>'
            )
        trs.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<div style="overflow-x:auto;overflow-y:auto;max-height:380px;'
        '-webkit-overflow-scrolling:touch;width:100%;">'
        '<table style="width:max-content;min-width:100%;border-collapse:collapse;">'
        f'<thead><tr>{"".join(ths)}</tr></thead>'
        f'<tbody>{"".join(trs)}</tbody>'
        '</table></div>'
    )


def _watchlist_df_html(portfolio_id: int) -> str:
    return _watchlist_html(_watchlist_df(portfolio_id), _watch_headers())


def _portfolio_tickers_str(portfolio_id: int = 1) -> str:
    """Comma-separated tickers from the active portfolio (for the RL tab)."""
    db = SessionLocal()
    try:
        tickers = [
            h.ticker
            for h in db.query(HoldingDB)
            .filter(HoldingDB.portfolio_id == portfolio_id)
            .all()
        ]
        return ", ".join(tickers)
    finally:
        db.close()


# ── Dashboard ─────────────────────────────────────────────────────────────────

def _period_date_range(period: str) -> str:
    """Return 'Mon DD, YYYY — Mon DD, YYYY' for a yfinance period string."""
    today = date.today()
    _map = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730, "5y": 1825, "10y": 3650}
    days = _map.get(period, 365)
    start = today - timedelta(days=days)
    return f"{start.strftime('%b %d, %Y')} — {today.strftime('%b %d, %Y')}"


def _portfolio_vs_spy_fig(
    portfolio_id: int,
    period: str = "1y",
    opt_date_override: "pd.Timestamp | None" = None,
) -> go.Figure:
    """Line chart: equal-weight & optimized-weight portfolio cumulative return vs ^GSPC."""
    import json
    from core.models import PortfolioAllocationDB

    with SessionLocal() as s:
        tickers = [h.ticker for h in
                   s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()]
        alloc_row = s.get(PortfolioAllocationDB, portfolio_id)

    # Parse optimized weights (stock-only, normalized so they sum to 1)
    opt_weights: dict[str, float] = {}
    opt_date: "pd.Timestamp | None" = opt_date_override
    if alloc_row:
        allocs = json.loads(alloc_row.allocations_json)
        total_w = sum(v["weight"] for v in allocs.values())
        if total_w > 0:
            opt_weights = {t: v["weight"] / total_w for t, v in allocs.items()}
        if opt_date is None and alloc_row.created_at:
            opt_date = pd.Timestamp(alloc_row.created_at.date())

    fig = go.Figure()
    fig.update_layout(
        title=f"Portfolio vs S&P 500 ({_period_date_range(period)})",
        template="plotly_dark",
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#111827",
        xaxis_title="Date",
        xaxis_range=[str(date.today() - timedelta(days=365)), str(date.today())],
        yaxis_title="Cumulative Return (%)",
        yaxis_ticksuffix="%",
        legend=dict(orientation="h", y=-0.2),
        height=460,
        autosize=True,
        margin=dict(l=48, r=24, t=48, b=60),
    )

    closes: dict[str, pd.Series] = {}
    spy = pd.Series(dtype=float)

    if tickers:
        import yfinance as yf
        symbols = list(dict.fromkeys(tickers + ["^GSPC"]))
        try:
            _end = date.today() + timedelta(days=1)
            _start = date.today() - timedelta(days=366)
            raw = yf.download(symbols, start=str(_start), end=str(_end),
                              group_by="ticker", auto_adjust=True,
                              threads=False, progress=False)
        except Exception as e:
            log.warning("vs-SPY: batched download failed: %s", e)
            raw = None

        if raw is not None and not raw.empty:
            for t in symbols:
                try:
                    series = raw[t]["Close"].dropna() if len(symbols) > 1 else raw["Close"].dropna()
                except Exception:
                    series = pd.Series(dtype=float)
                if t == "^GSPC":
                    spy = series
                elif not series.empty:
                    closes[t] = series
        log.info("vs-SPY: tickers=%d, rows per ticker=%s, spy_rows=%d",
                 len(tickers),
                 {t: len(s) for t, s in closes.items()},
                 len(spy))

    def _add_split_trace(
        x: "pd.DatetimeIndex",
        y: "np.ndarray",
        name: str,
        color: str,
        dash_before: str = "dot",
        width_before: float = 1.5,
        width_after: float = 2.5,
        opacity_before: float = 0.4,
        group: str = "",
    ) -> None:
        """Add one logical series as two styled segments split at opt_date."""
        # Split only when there's a meaningful "after" window (at least 2 points after opt_date)
        n_after = int((x > opt_date).sum()) if opt_date is not None else 0
        split = opt_date is not None and x[0] < opt_date and n_after >= 2
        if split:
            before = x <= opt_date
            after  = x >= opt_date  # overlap at opt_date keeps the line connected
            fig.add_trace(go.Scatter(
                x=x[before], y=y[before],
                mode="lines", name=name,
                line=dict(color=color, width=width_before, dash=dash_before),
                opacity=opacity_before,
                legendgroup=group, showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=x[after], y=y[after],
                mode="lines", name=name,
                line=dict(color=color, width=width_after),
                legendgroup=group, showlegend=True,
            ))
        else:
            # No split: render full line solid/bright (either no opt_date, or optimized today
            # with no post-optimization history yet, or opt_date is outside the data window)
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="lines", name=name,
                line=dict(color=color, width=width_after),
                legendgroup=group,
            ))

    if closes:
        df = pd.concat(closes, axis=1).dropna(how="any")
        if len(df) < 5:
            log.warning("vs-SPY: only %d aligned rows after dropna — insufficient", len(df))
            df = df.iloc[0:0]
        if not df.empty:
            norm = df.div(df.iloc[0]).sub(1).mul(100.0)

            port_eq = norm.mean(axis=1)
            _add_split_trace(port_eq.index, port_eq.values,
                             "Portfolio (equal-weighted)", "#00D4FF", group="eq")

            if opt_weights:
                common = [t for t in df.columns if t in opt_weights]
                if common:
                    w = pd.Series({t: opt_weights[t] for t in common})
                    w = w / w.sum()
                    opt_port = norm[common].mul(w, axis=1).sum(axis=1)
                    _add_split_trace(opt_port.index, opt_port.values,
                                     "Portfolio (optimized)", "#00FF94", group="opt")

    if not spy.empty:
        spy_norm = spy.div(spy.iloc[0]).sub(1).mul(100.0)
        _add_split_trace(spy_norm.index, spy_norm.values,
                         "S&P 500", "#FFA500", group="spy")

    # Vertical marker + shaded region for the post-optimization period
    if opt_date is not None and fig.data:
        all_x = pd.DatetimeIndex([t for tr in fig.data if tr.x is not None for t in tr.x])
        # Show marker whenever opt_date is within the data window (inclusive on both ends)
        if not all_x.empty and all_x.min() <= opt_date:
            marker_x = min(opt_date, all_x.max())
            fig.add_vline(
                x=str(marker_x.date()),
                line=dict(color="#FFD700", width=1.5, dash="dash"),
            )
            fig.add_annotation(
                x=marker_x, y=1.0, xref="x", yref="paper",
                text="<b>Optimized</b>", showarrow=False,
                font=dict(color="#FFD700", size=11),
                bgcolor="rgba(10,14,26,0.7)", bordercolor="#FFD700",
                borderwidth=1, borderpad=3,
                xanchor="left", yanchor="top", xshift=6,
            )
            if marker_x < all_x.max():
                fig.add_vrect(
                    x0=marker_x, x1=all_x.max(),
                    fillcolor="rgba(0,255,148,0.04)",
                    layer="below", line_width=0,
                )

    if not fig.data:
        fig.add_annotation(text="No price history available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#9CA3AF"))
    return fig


def _portfolio_vs_spy_sharpe_fig(
    portfolio_id: int,
    period: str = "1y",
    opt_date_override: "pd.Timestamp | None" = None,
    window: int = 63,
) -> go.Figure:
    """Rolling Sharpe ratio for equal-weight & optimized-weight portfolio vs S&P 500."""
    import json
    from core.models import PortfolioAllocationDB

    with SessionLocal() as s:
        tickers = [h.ticker for h in
                   s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()]
        alloc_row = s.get(PortfolioAllocationDB, portfolio_id)

    opt_weights: dict[str, float] = {}
    opt_date: "pd.Timestamp | None" = opt_date_override
    rf = 0.04
    if alloc_row:
        rf = float(alloc_row.risk_free_rate or 0.04)
        allocs = json.loads(alloc_row.allocations_json)
        total_w = sum(v["weight"] for v in allocs.values())
        if total_w > 0:
            opt_weights = {t: v["weight"] / total_w for t, v in allocs.items()}
        if opt_date is None and alloc_row.created_at:
            opt_date = pd.Timestamp(alloc_row.created_at.date())

    fig = go.Figure()
    fig.update_layout(
        title=f"Portfolio vs S&P 500 — Rolling Sharpe ({window}d, {_period_date_range(period)})",
        template="plotly_dark",
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#111827",
        xaxis=dict(
            title="Date",
            range=[str(date.today() - timedelta(days=365)), str(date.today())],
            autorange=False,
        ),
        yaxis_title="Rolling Sharpe Ratio (annualized)",
        legend=dict(orientation="h", y=-0.2),
        height=460,
        autosize=True,
        margin=dict(l=48, r=24, t=48, b=60),
    )

    closes: dict[str, pd.Series] = {}
    spy = pd.Series(dtype=float)

    if tickers:
        import yfinance as yf
        symbols = list(dict.fromkeys(tickers + ["^GSPC"]))
        try:
            _end = date.today() + timedelta(days=1)
            _start = date.today() - timedelta(days=366)
            raw = yf.download(symbols, start=str(_start), end=str(_end),
                              group_by="ticker", auto_adjust=True,
                              threads=False, progress=False)
        except Exception as e:
            log.warning("port-sharpe-vs-SPY: download failed: %s", e)
            raw = None

        if raw is not None and not raw.empty:
            for t in symbols:
                try:
                    series = raw[t]["Close"].dropna() if len(symbols) > 1 else raw["Close"].dropna()
                except Exception:
                    series = pd.Series(dtype=float)
                if t == "^GSPC":
                    spy = series
                elif not series.empty:
                    closes[t] = series

    def _rolling_sharpe(r: pd.Series) -> pd.Series:
        rm      = r.rolling(window).mean()
        rs      = r.rolling(window).std()
        ann_ret = rm * 252
        ann_vol = rs * np.sqrt(252)
        return ((ann_ret - rf) / ann_vol.where(ann_vol > 0, other=np.nan)).dropna()

    def _add_split_trace(x, y, name, color, group=""):
        n_after = int((x > opt_date).sum()) if opt_date is not None else 0
        split = opt_date is not None and len(x) > 0 and x[0] < opt_date and n_after >= 2
        if split:
            before = x <= opt_date
            after  = x >= opt_date
            fig.add_trace(go.Scatter(
                x=x[before], y=y[before], mode="lines", name=name,
                line=dict(color=color, width=1.5, dash="dot"),
                opacity=0.4, legendgroup=group, showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=x[after], y=y[after], mode="lines", name=name,
                line=dict(color=color, width=2.5),
                legendgroup=group, showlegend=True,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name=name,
                line=dict(color=color, width=2.5),
                legendgroup=group,
            ))

    if closes:
        df = pd.concat(closes, axis=1).dropna(how="any")
        if len(df) >= window + 5:
            rets = df.pct_change()

            eq_ret = rets.mean(axis=1)
            rs_eq = _rolling_sharpe(eq_ret)
            if not rs_eq.empty:
                _add_split_trace(rs_eq.index, rs_eq.values,
                                 "Portfolio (equal-weighted)", "#00D4FF", group="eq")

            if opt_weights:
                common = [t for t in df.columns if t in opt_weights]
                if common:
                    w = pd.Series({t: opt_weights[t] for t in common})
                    w = w / w.sum()
                    opt_ret = rets[common].mul(w, axis=1).sum(axis=1)
                    rs_opt = _rolling_sharpe(opt_ret)
                    if not rs_opt.empty:
                        _add_split_trace(rs_opt.index, rs_opt.values,
                                         "Portfolio (optimized)", "#00FF94", group="opt")

    if not spy.empty:
        rs_spy = _rolling_sharpe(spy.pct_change().dropna())
        if not rs_spy.empty:
            _add_split_trace(rs_spy.index, rs_spy.values, "S&P 500", "#FFA500", group="spy")

    # Yellow vertical line at optimization date
    if opt_date is not None and fig.data:
        all_x = pd.DatetimeIndex([t for tr in fig.data if tr.x is not None for t in tr.x])
        if not all_x.empty and all_x.min() <= opt_date:
            marker_x = min(opt_date, all_x.max())
            fig.add_vline(
                x=str(marker_x.date()),
                line=dict(color="#FFD700", width=1.5, dash="dash"),
            )
            fig.add_annotation(
                x=marker_x, y=1.0, xref="x", yref="paper",
                text="<b>Optimized</b>", showarrow=False,
                font=dict(color="#FFD700", size=11),
                bgcolor="rgba(10,14,26,0.7)", bordercolor="#FFD700",
                borderwidth=1, borderpad=3,
                xanchor="left", yanchor="top", xshift=6,
            )
            if marker_x < all_x.max():
                fig.add_vrect(
                    x0=marker_x, x1=all_x.max(),
                    fillcolor="rgba(0,255,148,0.04)",
                    layer="below", line_width=0,
                )

    if not fig.data:
        fig.add_annotation(text="No price history available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#9CA3AF"))
    return fig


# ── Color palette for individual stock traces ─────────────────────────────────
_STOCK_COLORS = [
    "#00D4FF", "#FF6B6B", "#FFD700", "#A78BFA", "#FF9F43",
    "#C8A2C8", "#7ED321", "#9B59B6", "#E74C3C", "#3498DB",
    "#1ABC9C", "#F39C12", "#2ECC71", "#E91E63", "#FF5722",
]


def _stocks_vs_spy_return_fig(
    portfolio_id: int,
    period: str = "1y",
    opt_date_override: "pd.Timestamp | None" = None,
) -> go.Figure:
    """Cumulative return % for each individual stock vs S&P 500."""
    import json
    from core.models import PortfolioAllocationDB

    with SessionLocal() as s:
        tickers = [h.ticker for h in
                   s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()]
        alloc_row = s.get(PortfolioAllocationDB, portfolio_id)

    opt_tickers: set[str] = set()
    opt_date: "pd.Timestamp | None" = opt_date_override
    if alloc_row:
        allocs = json.loads(alloc_row.allocations_json)
        opt_tickers = set(allocs.keys())
        if opt_date is None and alloc_row.created_at:
            opt_date = pd.Timestamp(alloc_row.created_at.date())

    fig = go.Figure()
    fig.update_layout(
        title=f"Individual Stocks vs S&P 500 — Return % ({_period_date_range(period)})",
        template="plotly_dark",
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#111827",
        xaxis=dict(
            title="Date",
            range=[str(date.today() - timedelta(days=365)), str(date.today())],
            autorange=False,
        ),
        yaxis_title="Cumulative Return (%)",
        yaxis_ticksuffix="%",
        legend=dict(orientation="h", y=-0.2),
        height=460,
        autosize=True,
        margin=dict(l=48, r=24, t=48, b=60),
    )

    if not tickers:
        fig.add_annotation(text="No holdings in portfolio",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#9CA3AF"))
        return fig

    import yfinance as yf
    symbols = list(dict.fromkeys(tickers + ["^GSPC"]))
    try:
        _end = date.today() + timedelta(days=1)
        _start = date.today() - timedelta(days=366)
        raw = yf.download(symbols, start=str(_start), end=str(_end),
                          group_by="ticker", auto_adjust=True,
                          threads=False, progress=False)
    except Exception as e:
        log.warning("stocks-vs-SPY-return: download failed: %s", e)
        raw = None

    closes: dict[str, pd.Series] = {}
    if raw is not None and not raw.empty:
        for t in symbols:
            try:
                series = raw[t]["Close"].dropna() if len(symbols) > 1 else raw["Close"].dropna()
            except Exception:
                series = pd.Series(dtype=float)
            if not series.empty:
                closes[t] = series

    if not closes:
        fig.add_annotation(text="No price history available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#9CA3AF"))
        return fig

    df = pd.concat({k: v for k, v in closes.items()}, axis=1).dropna(how="any")
    if len(df) < 5:
        fig.add_annotation(text="Insufficient price history",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#9CA3AF"))
        return fig

    norm = df.div(df.iloc[0]).sub(1).mul(100.0)

    def _add_split_trace_r(x, y, name, color, group=""):
        n_after = int((x > opt_date).sum()) if opt_date is not None else 0
        split = opt_date is not None and len(x) > 0 and x[0] < opt_date and n_after >= 2
        if split:
            before = x <= opt_date
            after  = x >= opt_date
            fig.add_trace(go.Scatter(
                x=x[before], y=y[before], mode="lines", name=name,
                line=dict(color=color, width=1.5, dash="dot"),
                opacity=0.4, legendgroup=group, showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=x[after], y=y[after], mode="lines", name=name,
                line=dict(color=color, width=2.5),
                legendgroup=group, showlegend=True,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name=name,
                line=dict(color=color, width=2.5),
                legendgroup=group,
            ))

    stock_cols = [c for c in df.columns if c != "^GSPC"]
    for i, ticker in enumerate(stock_cols):
        color = _STOCK_COLORS[i % len(_STOCK_COLORS)]
        _add_split_trace_r(norm.index, norm[ticker].values, ticker, color, group=ticker)

    if "^GSPC" in norm.columns:
        _add_split_trace_r(norm.index, norm["^GSPC"].values, "S&P 500", "#FFA500", group="spy")

    # Yellow vertical line at optimization date
    if opt_date is not None and fig.data:
        all_x = pd.DatetimeIndex([t for tr in fig.data if tr.x is not None for t in tr.x])
        if not all_x.empty and all_x.min() <= opt_date:
            marker_x = min(opt_date, all_x.max())
            fig.add_vline(
                x=str(marker_x.date()),
                line=dict(color="#FFD700", width=1.5, dash="dash"),
            )
            fig.add_annotation(
                x=marker_x, y=1.0, xref="x", yref="paper",
                text="<b>Optimized</b>", showarrow=False,
                font=dict(color="#FFD700", size=11),
                bgcolor="rgba(10,14,26,0.7)", bordercolor="#FFD700",
                borderwidth=1, borderpad=3,
                xanchor="left", yanchor="top", xshift=6,
            )
            if marker_x < all_x.max():
                fig.add_vrect(
                    x0=marker_x, x1=all_x.max(),
                    fillcolor="rgba(0,255,148,0.04)",
                    layer="below", line_width=0,
                )

    return fig


def _stocks_vs_spy_sharpe_fig(
    portfolio_id: int,
    period: str = "1y",
    opt_date_override: "pd.Timestamp | None" = None,
    window: int = 63,
) -> go.Figure:
    """Rolling Sharpe ratio (annualized) for each individual stock vs S&P 500."""
    import json
    from core.models import PortfolioAllocationDB

    with SessionLocal() as s:
        tickers = [h.ticker for h in
                   s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()]
        alloc_row = s.get(PortfolioAllocationDB, portfolio_id)

    opt_date: "pd.Timestamp | None" = opt_date_override
    rf = 0.04
    if alloc_row:
        rf = float(alloc_row.risk_free_rate or 0.04)
        if opt_date is None and alloc_row.created_at:
            opt_date = pd.Timestamp(alloc_row.created_at.date())

    fig = go.Figure()
    fig.update_layout(
        title=f"Individual Stocks vs S&P 500 — Rolling Sharpe ({window}d, {_period_date_range(period)})",
        template="plotly_dark",
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#111827",
        xaxis=dict(
            title="Date",
            range=[str(date.today() - timedelta(days=365)), str(date.today())],
            autorange=False,
        ),
        yaxis_title="Rolling Sharpe Ratio (annualized)",
        legend=dict(orientation="h", y=-0.2),
        height=460,
        autosize=True,
        margin=dict(l=48, r=24, t=48, b=60),
    )

    if not tickers:
        fig.add_annotation(text="No holdings in portfolio",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#9CA3AF"))
        return fig

    import yfinance as yf
    symbols = list(dict.fromkeys(tickers + ["^GSPC"]))
    try:
        _end = date.today() + timedelta(days=1)
        _start = date.today() - timedelta(days=366)
        raw = yf.download(symbols, start=str(_start), end=str(_end),
                          group_by="ticker", auto_adjust=True,
                          threads=False, progress=False)
    except Exception as e:
        log.warning("stocks-vs-SPY-sharpe: download failed: %s", e)
        raw = None

    closes: dict[str, pd.Series] = {}
    if raw is not None and not raw.empty:
        for t in symbols:
            try:
                series = raw[t]["Close"].dropna() if len(symbols) > 1 else raw["Close"].dropna()
            except Exception:
                series = pd.Series(dtype=float)
            if not series.empty:
                closes[t] = series

    if not closes:
        fig.add_annotation(text="No price history available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#9CA3AF"))
        return fig

    df = pd.concat({k: v for k, v in closes.items()}, axis=1).dropna(how="any")
    if len(df) < window + 5:
        fig.add_annotation(
            text=f"Need at least {window + 5} trading days of history for rolling Sharpe",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#9CA3AF"))
        return fig

    rets = df.pct_change()

    def _rolling_sharpe(r: pd.Series) -> pd.Series:
        # Full window required — early partial-window Sharpe is statistically
        # unreliable (std estimate has large uncertainty below ~63 samples).
        rm  = r.rolling(window).mean()
        rs  = r.rolling(window).std()
        ann_ret = rm * 252
        ann_vol = rs * np.sqrt(252)
        sharpe  = (ann_ret - rf) / ann_vol.where(ann_vol > 0, other=np.nan)
        return sharpe.dropna()

    def _add_split_trace_s(x, y, name, color, group=""):
        n_after = int((x > opt_date).sum()) if opt_date is not None else 0
        split = opt_date is not None and len(x) > 0 and x[0] < opt_date and n_after >= 2
        if split:
            before = x <= opt_date
            after  = x >= opt_date
            fig.add_trace(go.Scatter(
                x=x[before], y=y[before], mode="lines", name=name,
                line=dict(color=color, width=1.5, dash="dot"),
                opacity=0.4, legendgroup=group, showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=x[after], y=y[after], mode="lines", name=name,
                line=dict(color=color, width=2.5),
                legendgroup=group, showlegend=True,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name=name,
                line=dict(color=color, width=2.5),
                legendgroup=group,
            ))

    stock_cols = [c for c in df.columns if c != "^GSPC"]
    for i, ticker in enumerate(stock_cols):
        color = _STOCK_COLORS[i % len(_STOCK_COLORS)]
        rs = _rolling_sharpe(rets[ticker])
        if not rs.empty:
            _add_split_trace_s(rs.index, rs.values, ticker, color, group=ticker)

    if "^GSPC" in rets.columns:
        rs_spy = _rolling_sharpe(rets["^GSPC"])
        if not rs_spy.empty:
            _add_split_trace_s(rs_spy.index, rs_spy.values, "S&P 500", "#FFA500", group="spy")

    # Yellow vertical line at optimization date
    if opt_date is not None and fig.data:
        all_x = pd.DatetimeIndex([t for tr in fig.data if tr.x is not None for t in tr.x])
        if not all_x.empty and all_x.min() <= opt_date:
            marker_x = min(opt_date, all_x.max())
            fig.add_vline(
                x=str(marker_x.date()),
                line=dict(color="#FFD700", width=1.5, dash="dash"),
            )
            fig.add_annotation(
                x=marker_x, y=1.0, xref="x", yref="paper",
                text="<b>Optimized</b>", showarrow=False,
                font=dict(color="#FFD700", size=11),
                bgcolor="rgba(10,14,26,0.7)", bordercolor="#FFD700",
                borderwidth=1, borderpad=3,
                xanchor="left", yanchor="top", xshift=6,
            )
            if marker_x < all_x.max():
                fig.add_vrect(
                    x0=marker_x, x1=all_x.max(),
                    fillcolor="rgba(0,255,148,0.04)",
                    layer="below", line_width=0,
                )

    if not fig.data:
        fig.add_annotation(text="Insufficient data for rolling Sharpe",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#9CA3AF"))

    return fig


def refresh_dashboard(portfolio_id: int = 1):
    watch          = [r for r in live_watchlist_rows(portfolio_id) if not (len(r) > 8 and r[8] == "red")]
    vs_spy         = _portfolio_vs_spy_fig(portfolio_id)
    port_sharpe    = _portfolio_vs_spy_sharpe_fig(portfolio_id)
    stocks_return  = _stocks_vs_spy_return_fig(portfolio_id)
    stocks_sharpe  = _stocks_vs_spy_sharpe_fig(portfolio_id)
    rows, m        = last_plan_rows(portfolio_id)
    watch_lbl          = _date_range_label("Live watchlist")
    spy_lbl            = _date_range_label("Portfolio vs S&P 500")
    port_sharpe_lbl    = _date_range_label("Portfolio vs S&P 500 — Rolling Sharpe (63d)")
    stocks_return_lbl  = _date_range_label("Individual Stocks vs S&P 500 — Return %")
    stocks_sharpe_lbl  = _date_range_label("Individual Stocks vs S&P 500 — Rolling Sharpe (63d)")
    if m is None:
        return (_watchlist_html(watch, _watch_headers()),
                "—", "—", "—",
                gr.update(value="—", label="Sharpe (ann.)"),
                gr.update(value="—", label="Sortino (ann.)"),
                "—", "—", _placeholder(420),
                _watchlist_html([], _ALLOC_HEADERS),
                "_Run the Optimizer to see your plan._", vs_spy, watch_lbl, spy_lbl,
                port_sharpe, port_sharpe_lbl,
                stocks_return, stocks_return_lbl,
                stocks_sharpe, stocks_sharpe_lbl)
    sortino_s      = f"{m['sortino']:.3f}" if m.get("sortino") is not None else "—"
    var_s          = f"{m['var_95']*100:.2f}%" if m.get("var_95") is not None else "—"
    opt_date_raw   = m.get("opt_date") or "unknown"
    return (
        _watchlist_html(watch, _watch_headers()),
        f"${m['budget']:,.0f}",
        f"{m['expected_return']*100:.2f}%",
        f"{m['expected_vol']*100:.2f}%",
        gr.update(value=f"{m['sharpe']:.3f}", label=f"Sharpe (ann.) at {opt_date_raw}"),
        gr.update(value=f"{sortino_s}",       label=f"Sortino (ann.) at {opt_date_raw}"),
        var_s,
        f"${m['cash_dollars']:,.0f}",
        last_plan_pie(portfolio_id),
        _watchlist_html(rows, _ALLOC_HEADERS),
        f"_Last optimized: {opt_date_raw}_",
        vs_spy,
        watch_lbl,
        spy_lbl,
        port_sharpe,
        port_sharpe_lbl,
        stocks_return,
        stocks_return_lbl,
        stocks_sharpe,
        stocks_sharpe_lbl,
    )


# ── Positions helpers ─────────────────────────────────────────────────────────

# ── Portfolio CRUD ─────────────────────────────────────────────────────────────

def add_ticker(ticker: str, portfolio_id: int):
    try:
        payload = HoldingCreate(ticker=ticker)
    except ValidationError as ve:
        msg = ve.errors()[0].get("msg", "invalid ticker")
        return gr.update(), f"❌ {msg}", gr.update(), gr.update(), _watchlist_df_html(portfolio_id)
    except Exception as e:
        return gr.update(), f"❌ {e}", gr.update(), gr.update(), _watchlist_df_html(portfolio_id)
    with SessionLocal() as s:
        exists = (s.query(HoldingDB)
                   .filter_by(portfolio_id=portfolio_id, ticker=payload.ticker)
                   .first())
        if exists:
            return (gr.update(value=""),
                    f"ℹ️ {payload.ticker} already on watchlist",
                    gr.update(), gr.update(),
                    _watchlist_df_html(portfolio_id))
        s.add(HoldingDB(portfolio_id=portfolio_id, ticker=payload.ticker))
        s.commit()
    rows = _watchlist_df(portfolio_id)
    dd = gr.update(choices=_all_tickers(portfolio_id), value=None)
    return gr.update(value=""), f"✅ Added {payload.ticker}", dd, dd, _watchlist_html(rows, _watch_headers())


def update_position(ticker: str, shares, purchase_price, portfolio_id: int):
    if not ticker or ticker == "CASH":
        return "⚠️ Select a ticker to edit", gr.update()
    sh = float(shares) if shares else None
    pp = float(purchase_price) if purchase_price else None
    with SessionLocal() as s:
        row = s.query(HoldingDB).filter_by(
            portfolio_id=portfolio_id, ticker=ticker).first()
        if not row:
            return f"❌ {ticker} not found", gr.update()
        if sh is not None:
            row.shares = sh
        if pp is not None:
            row.purchase_price = pp
        s.commit()
    return f"✅ Updated {ticker}", _watchlist_df(portfolio_id)


def remove_ticker(ticker: str, portfolio_id: int):
    if not ticker or ticker == "CASH":
        rows = _watchlist_df(portfolio_id)
        return "⚠️ pick a ticker (CASH is not removable)", _watchlist_html(rows, _watch_headers()), gr.update(), gr.update()
    with SessionLocal() as s:
        row = (s.query(HoldingDB)
                .filter_by(portfolio_id=portfolio_id, ticker=ticker).first())
        if row:
            s.delete(row)
            s.commit()
    rows = _watchlist_df(portfolio_id)
    dd = gr.update(choices=_all_tickers(portfolio_id), value=None)
    return f"🗑️ Removed {ticker}", _watchlist_html(rows, _watch_headers()), dd, dd


# ── Chat handler ───────────────────────────────────────────────────────────────

def handle_chat(message, history, tts_on, portfolio_id: int = 1):
    if not message.strip():
        yield (gr.update(value="", interactive=True), history, "", "",
               _EMPTY, _EMPTY, _EMPTY, _EMPTY, gr.update(visible=False), "")
        return

    user_msg = {"role": "user", "content": message}
    pending = history + [user_msg, {"role": "assistant", "content": "⏳ Agents working…"}]
    yield (gr.update(value="⏳ Thinking…", interactive=False),
           pending, "", "", _EMPTY, _EMPTY, _EMPTY, _EMPTY, gr.update(visible=False), "")

    response, charts, agents_used, status_log = run_agents(message, history, portfolio_id)
    new_history = history + [user_msg, {"role": "assistant", "content": response}]
    badges_html = agent_badges_html(agents_used)
    audio_html  = tts_html(response, tts_on)

    def _fig(i):
        return charts[i] if i < len(charts) else _EMPTY

    yield (
        gr.update(value="", interactive=True),
        new_history, badges_html, audio_html,
        _fig(0), _fig(1), _fig(2), _fig(3),
        gr.update(visible=bool(charts)),
        tts_text_for_js(response),
    )


# ── Portfolio-switch handler (refreshes all tabs) ─────────────────────────────

def _load_saved_optimizer(pid: int) -> tuple:
    """Return 19-element tuple: 11 optimizer outputs + 8 input param updates."""
    import json
    from core.database import SessionLocal
    from core.models import PortfolioAllocationDB
    from services.optimizer import build_plots

    _defaults = (
        100_000,   # budget
        15.0,      # target_vol slider (%)
        4.00,      # rf_slider (%)
        "4.00%",   # rf_text
        "2y",      # lookback
        5_000,     # frontier
        1.0,       # sr_slider
        "1.00",    # sr_text
    )

    with SessionLocal() as s:
        row = s.get(PortfolioAllocationDB, pid)
        if row is None:
            return (
                "", "", "",
                gr.update(value="", label="Sharpe (ann.)"),
                gr.update(value="", label="Sortino (ann.)"),
                "", "", None, None, None, "",
            ) + _defaults
        allocs_json      = row.allocations_json
        frontier_json    = getattr(row, 'frontier_json', None) or "[]"
        cash_dollars     = row.cash_dollars
        commentary       = getattr(row, 'commentary', None) or ""
        budget           = row.budget
        target_vol_pct   = row.target_vol * 100.0
        rf_pct           = row.risk_free_rate * 100.0
        lookback         = row.lookback
        frontier_samples = int(getattr(row, 'frontier_samples', None) or 5_000)
        sr_threshold     = float(getattr(row, 'sr_threshold', None) or 1.0)
        created_at       = row.created_at
        metrics = {
            "expected_return": row.expected_return,
            "expected_vol":    row.expected_vol,
            "sharpe":          row.sharpe,
            "sortino":         row.sortino,
            "var_95":          row.var_95,
        }

    allocs = json.loads(allocs_json)

    result = {
        "allocations":     allocs,
        "cash_dollars":    cash_dollars,
        "metrics":         metrics,
        "frontier_points": json.loads(frontier_json),
        "warnings":        [],
    }
    fig_p, fig_b, fig_f = build_plots(result)
    sortino_s    = f"{metrics['sortino']:.3f}" if metrics.get("sortino") is not None else "—"
    var_s        = f"{metrics['var_95']*100:.2f}%" if metrics.get("var_95") is not None else "—"
    opt_date_str = getattr(row, "opt_date", None) or (
        created_at.strftime("%Y-%m-%d UTC") if created_at else "unknown"
    )
    return (
        "✅ Last saved",
        f"{metrics['expected_return']*100:.2f}%",
        f"{metrics['expected_vol']*100:.2f}%",
        gr.update(value=f"{metrics['sharpe']:.3f}", label="Sharpe (ann.)",  info=f"at {opt_date_str}"),
        gr.update(value=sortino_s,                  label="Sortino (ann.)", info=f"at {opt_date_str}"),
        var_s,
        f"${cash_dollars:,.0f}",
        fig_p, fig_b, fig_f,
        commentary,
        # ── input param restores ──────────────────────────────────────
        budget,
        target_vol_pct,
        rf_pct,
        f"{rf_pct:.2f}%",
        lookback,
        frontier_samples,
        sr_threshold,
        f"{sr_threshold:.2f}",
    )


def _switch_portfolio(choice: str):
    pid  = _id_from_choice(choice)
    dash = refresh_dashboard(pid)
    rows = _watchlist_df(pid)
    tickers = [r[0] for r in rows if r[0] != "CASH"]
    dd = gr.update(choices=tickers, value=None)
    opt = _load_saved_optimizer(pid)
    return (
        pid, *dash, _watchlist_html(rows, _watch_headers()), dd, dd,
        _date_range_label("Live prices"),
        [],   # chatbot: clear history
        *opt,
    )


# ── Interface factory ─────────────────────────────────────────────────────────

def create_interface(theme=None, css: str | None = None, js: str | None = None) -> gr.Blocks:
    init_db()
    initial_choices = _portfolio_choices()
    initial_choice  = initial_choices[0] if initial_choices else None

    with gr.Blocks(title="AI Portfolio Manager", analytics_enabled=False,
                   theme=theme, css=css, js=js) as demo:
        gr.HTML(_runtime_banner_html())

        # Shared portfolio state (holds active portfolio ID integer)
        portfolio_state = gr.State(_id_from_choice(initial_choice))

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:24px 0 8px;">
          <h1 class="app-title">📈 AI Portfolio Manager</h1>
          <p style="color:#9CA3AF;margin:6px 0 0;font-size:.95rem;">
            LangGraph Multi-Agent &nbsp;·&nbsp; Markowitz-Optimised &nbsp;·&nbsp;
            RAG Knowledge Base &nbsp;·&nbsp; Real-time Prices
          </p>
        </div>
        """)

        # ── Portfolio Selector Bar ─────────────────────────────────────────────
        with gr.Group(elem_classes="portfolio-bar"):
            with gr.Row():
                pf_drop = gr.Dropdown(
                    choices=initial_choices,
                    value=initial_choice,
                    label="📁 Active Portfolio",
                    interactive=True,
                    scale=3,
                )
                pf_new_name = gr.Textbox(
                    placeholder="New portfolio name…",
                    show_label=False, scale=2,
                )
                pf_create_btn = gr.Button(
                    "➕ Create", variant="primary", scale=1, min_width=90,
                    elem_classes="btn-glow",
                )
                pf_rename_name = gr.Textbox(
                    placeholder="Rename active to…",
                    show_label=False, scale=2,
                )
                pf_rename_btn = gr.Button(
                    "✏️ Rename", variant="secondary", scale=1, min_width=90,
                )
                pf_delete_btn = gr.Button(
                    "🗑️ Delete", variant="stop", scale=1, min_width=90,
                )
            pf_status = gr.Markdown(value="")

        # ── Agent Workflow Diagram (top-level toggle) ──────────────────────────
        def _build_workflow_html(on: bool):
            if not on:
                return gr.update(value="", visible=False)
            try:
                import base64
                from agents.graph import get_graph
                png_bytes = get_graph().get_graph(xray=True).draw_mermaid_png()
                b64 = base64.b64encode(png_bytes).decode()
                html = (
                    '<div class="mermaid-wrap">'
                    f'<img src="data:image/png;base64,{b64}" '
                    'style="max-width:100%;border-radius:8px;">'
                    '</div>'
                )
            except Exception:
                html = _MERMAID_HTML
            return gr.update(value=html, visible=True)

        with gr.Row():
            show_workflow = gr.Checkbox(
                label="🔀 Show Agent Workflow Diagram",
                value=False, interactive=True,
            )
        workflow_diagram = gr.HTML(value="", visible=False, elem_classes="mermaid-wrap")
        show_workflow.change(
            fn=_build_workflow_html,
            inputs=show_workflow,
            outputs=workflow_diagram,
        )

        with gr.Tabs(elem_classes="tab-nav"):

            # ════════════════════════════════════════════════════════════════
            # TAB 1 — Dashboard
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("📊 Dashboard") as dash_tab:
                watch_label = gr.Markdown(_date_range_label("Live watchlist"))
                dash_watch = gr.HTML(value="")
                gr.Markdown("### Last optimized plan")
                with gr.Row():
                    d_budget  = gr.Textbox(label="Budget",         interactive=False)
                    d_ret     = gr.Textbox(label="Expected return", interactive=False)
                    d_vol     = gr.Textbox(label="Expected vol",    interactive=False)
                with gr.Row():
                    d_shrp    = gr.Textbox(label="Sharpe (ann.)",   interactive=False, info="")
                    d_sortino = gr.Textbox(label="Sortino (ann.)",  interactive=False, info="")
                    d_var     = gr.Textbox(label="VaR 95% (ann.)",  interactive=False)
                    d_cash    = gr.Textbox(label="Cash",            interactive=False)
                dash_pie = gr.Plot(label="Allocation", min_width=400, value=_placeholder(420))
                dash_table = gr.HTML(value="")
                d_stamp = gr.Markdown()
                spy_label = gr.Markdown(_date_range_label("Portfolio vs S&P 500"))
                dash_vs_spy = gr.Plot(label="Portfolio vs S&P 500", min_width=400, value=_placeholder(460))
                port_sharpe_label = gr.Markdown(_date_range_label("Portfolio vs S&P 500 — Rolling Sharpe (63d)"))
                dash_port_sharpe = gr.Plot(label="Portfolio vs S&P 500 — Rolling Sharpe (63d)", min_width=400, value=_placeholder(460))
                stocks_return_label = gr.Markdown(_date_range_label("Individual Stocks vs S&P 500 — Return %"))
                dash_stocks_return = gr.Plot(label="Individual Stocks vs S&P 500 — Return %", min_width=400, value=_placeholder(460))
                stocks_sharpe_label = gr.Markdown(_date_range_label("Individual Stocks vs S&P 500 — Rolling Sharpe (63d)"))
                dash_stocks_sharpe = gr.Plot(label="Individual Stocks vs S&P 500 — Rolling Sharpe (63d)", min_width=400, value=_placeholder(460))
                refresh_btn = gr.Button("🔄 Refresh dashboard", variant="primary",
                                        elem_classes="btn-glow")

                _dash_outs = [dash_watch, d_budget, d_ret, d_vol, d_shrp,
                              d_sortino, d_var, d_cash,
                              dash_pie, dash_table, d_stamp, dash_vs_spy,
                              watch_label, spy_label,
                              dash_port_sharpe, port_sharpe_label,
                              dash_stocks_return, stocks_return_label,
                              dash_stocks_sharpe, stocks_sharpe_label]
                refresh_btn.click(refresh_dashboard, [portfolio_state], _dash_outs)
                demo.load(refresh_dashboard, [portfolio_state], _dash_outs)

            # ════════════════════════════════════════════════════════════════
            # TAB 2 — Manage Holdings
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("📋 Portfolio"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Add ticker")
                        ticker_in  = gr.Textbox(label="Ticker (e.g. AAPL)", max_lines=1)
                        add_btn    = gr.Button("Add", variant="primary")
                        add_status = gr.Markdown()
                    with gr.Column(scale=1):
                        gr.Markdown("### Remove ticker")
                        remove_dd     = gr.Dropdown(label="Select to remove", choices=[])
                        remove_btn    = gr.Button("Remove", variant="stop")
                        remove_status = gr.Markdown()

                prices_label = gr.Markdown(_date_range_label("Live prices"))
                watchlist_df = gr.HTML(value="")

                # stub — kept for _switch_outs tuple compat
                edit_dd = gr.State(None)

                add_btn.click(
                    add_ticker,
                    inputs=[ticker_in, portfolio_state],
                    outputs=[ticker_in, add_status, remove_dd, edit_dd, watchlist_df],
                )
                remove_btn.click(
                    remove_ticker,
                    inputs=[remove_dd, portfolio_state],
                    outputs=[remove_status, watchlist_df, remove_dd, edit_dd],
                )

                def _init_watchlist(pid: int):
                    rows = _watchlist_df(pid)
                    dd = gr.update(choices=_all_tickers(pid), value=None)
                    return _watchlist_html(rows, _watch_headers()), dd, _date_range_label("Live prices")

                demo.load(
                    _init_watchlist,
                    [portfolio_state],
                    [watchlist_df, remove_dd, prices_label],
                )

            # ════════════════════════════════════════════════════════════════
            # TAB 3 — Optimizer (Markowitz + CASH, budget-based)
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("🤖 Optimizer"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=220):
                        opt_budget = gr.Number(
                            label="Budget ($)",
                            value=100_000, minimum=1_000, step=1_000,
                        )
                        opt_target_vol = gr.Slider(
                            label="Target risk (annual vol, %)",
                            minimum=5, maximum=40, value=15, step=0.5,
                        )
                        with gr.Row():
                            opt_rf_slider = gr.Slider(
                                label="Risk-free rate (%)",
                                minimum=0, maximum=20, value=4.00, step=0.25,
                            )
                            opt_rf_text = gr.Textbox(
                                label="…or type (e.g. 4.56%)",
                                value="4.00%", max_lines=1,
                            )
                        with gr.Row():
                            opt_sr_slider = gr.Slider(
                                label="SR screen threshold",
                                minimum=0.0, maximum=5.0, value=1.0, step=0.1,
                            )
                            opt_sr_text = gr.Textbox(
                                label="…or type (e.g. 1.5)",
                                value="1.00", max_lines=1,
                            )
                        opt_lookback = gr.Dropdown(
                            label="Lookback window",
                            choices=["1y", "2y", "3y", "5y"], value="2y",
                        )
                        opt_frontier = gr.Slider(
                            label="Portfolio samples (Monte Carlo cloud size)",
                            minimum=2_000, maximum=999_999, value=5_000, step=1_000,
                        )
                        opt_btn = gr.Button("Optimize", variant="primary")
                    with gr.Column(scale=3):
                        opt_status = gr.Markdown()
                        with gr.Row():
                            m_ret     = gr.Textbox(label="Expected return",  interactive=False)
                            m_vol     = gr.Textbox(label="Expected vol",     interactive=False)
                        with gr.Row():
                            m_shrp    = gr.Textbox(label="Sharpe (ann.)",    interactive=False, info="")
                            m_sortino = gr.Textbox(label="Sortino (ann.)",   interactive=False, info="")
                            m_var     = gr.Textbox(label="VaR 95% (ann.)",   interactive=False)
                            m_cash    = gr.Textbox(label="Cash reserve ($)",  interactive=False)
                        fig_pie      = gr.Plot(label="Allocation")
                        fig_bar      = gr.Plot(label="Dollar allocation")
                        fig_frontier = gr.Plot(label="Efficient frontier",
                                              elem_id="fig-frontier")
                        frontier_click_data = gr.Textbox(
                            value="",
                            elem_id="frontier-click-data",
                            show_label=False,
                        )
                        frontier_trigger_btn = gr.Button(
                            "", visible=False, elem_id="frontier-trigger-btn"
                        )
                        opt_commentary = gr.Markdown()

                opt_rf_slider.change(
                    sync_slider_to_text,
                    inputs=[opt_rf_slider], outputs=[opt_rf_text],
                )
                opt_rf_text.submit(
                    sync_text_to_slider,
                    inputs=[opt_rf_text, opt_rf_slider],
                    outputs=[opt_rf_slider, opt_rf_text],
                )
                opt_sr_slider.change(
                    sync_sr_slider_to_text,
                    inputs=[opt_sr_slider], outputs=[opt_sr_text],
                )
                opt_sr_text.submit(
                    sync_sr_text_to_slider,
                    inputs=[opt_sr_text, opt_sr_slider],
                    outputs=[opt_sr_slider, opt_sr_text],
                )
                opt_btn.click(
                    run_optimize,
                    inputs=[opt_budget, opt_target_vol, opt_rf_text,
                            opt_lookback, opt_frontier, portfolio_state, opt_sr_text],
                    outputs=[opt_status, m_ret, m_vol, m_shrp,
                             m_sortino, m_var, m_cash,
                             fig_pie, fig_bar, fig_frontier, opt_commentary,
                             opt_target_vol]
                            + _dash_outs,
                ).then(
                    _init_watchlist,
                    [portfolio_state],
                    [watchlist_df, remove_dd, prices_label],
                )

                # Belt-and-suspenders: wire both paths so the bridge works on
                # Gradio 6.8 (local, change event) and 6.9/HF Space (button click).
                # frontier_confirm deduplicates by timestamp to prevent double-firing.
                frontier_click_data.change(
                    frontier_confirm,
                    inputs=[frontier_click_data, opt_budget, opt_rf_text,
                            opt_lookback, opt_frontier, portfolio_state, opt_sr_text],
                    outputs=[opt_status, m_ret, m_vol, m_shrp,
                             m_sortino, m_var, m_cash,
                             fig_pie, fig_bar, fig_frontier, opt_commentary,
                             opt_target_vol]
                            + _dash_outs,
                ).then(
                    _init_watchlist,
                    [portfolio_state],
                    [watchlist_df, remove_dd, prices_label],
                )
                frontier_trigger_btn.click(
                    frontier_confirm,
                    inputs=[frontier_click_data, opt_budget, opt_rf_text,
                            opt_lookback, opt_frontier, portfolio_state, opt_sr_text],
                    outputs=[opt_status, m_ret, m_vol, m_shrp,
                             m_sortino, m_var, m_cash,
                             fig_pie, fig_bar, fig_frontier, opt_commentary,
                             opt_target_vol]
                            + _dash_outs,
                ).then(
                    _init_watchlist,
                    [portfolio_state],
                    [watchlist_df, remove_dd, prices_label],
                )

            # ════════════════════════════════════════════════════════════════
            # TAB 4 — AI Assistant (LangGraph multi-agent)
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("💬 AI Assistant"):
                gr.HTML(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                  <div>
                    <strong style="color:#00D4FF;font-size:1.1rem;">
                      Multi-Agent Portfolio Assistant
                    </strong>
                    <span style="color:#6B7280;font-size:.85rem;margin-left:12px;">
                      LangGraph · {llm_display_name()}
                    </span>
                  </div>
                </div>
                <p style="color:#9CA3AF;font-size:.82rem;margin:4px 0 0;">
                  Ask anything — supervisor routes to specialist agents automatically.<br>
                  Say <em>"rebalance my portfolio with $50k"</em> → Optimizer triggers automatically.
                </p>
                """)

                chatbot = gr.Chatbot(
                    label="", height=650,
                    elem_id="main-chatbot",
                    avatar_images=(
                        None,
                        "https://cdn-icons-png.flaticon.com/512/4616/4616734.png",
                    ),
                )

                badges_out     = gr.HTML(value="", visible=True)

                # ── Sample questions (3 rows × 4) ─────────────────────────────
                _SAMPLE_QS = [
                    "How you can help me?",
                    "what are my holdings?",
                    "What are my today's Profit/Loss?",
                    "Which stock is my best performer?",
                    "Analyze my portfolio volatility",
                    "Compare my portfolio vs S&P 500",
                    "What is my portfolio risk exposure?",
                    "What is Value at Risk (VaR)?",
                    "Rebalance for maximum Sharpe ratio",
                    "Rebalance for a Sharpe ratio of 1.5",
                    "Explain what Sharpe ratio means",
                    "Explain what Sortino ratio means",
                ]
                _sample_btns = []
                gr.HTML('<p style="color:#6B7280;font-size:.78rem;margin:8px 0 4px;">💡 Try asking:</p>')
                for _row_start in range(0, 12, 4):
                    with gr.Row():
                        for _q in _SAMPLE_QS[_row_start:_row_start + 4]:
                            _b = gr.Button(_q, size="sm", variant="secondary", scale=1)
                            _sample_btns.append((_b, _q))

                with gr.Row():
                    msg_box  = gr.Textbox(
                        placeholder=(
                            "Ask anything: 'How is my portfolio?' · "
                            "'Rebalance with $100k' · 'What is Sharpe ratio?'"
                        ),
                        show_label=False, scale=5, elem_id="chat-input",
                    )
                    send_btn = gr.Button(
                        "Send 📨", variant="primary", scale=1, elem_classes="btn-glow"
                    )

                with gr.Row():
                    tts_state = gr.State(False)
                    tts_btn   = gr.Button(
                        "🔊 Read", variant="secondary", scale=1,
                        elem_classes="tts-btn-off", elem_id="tts-btn",
                    )
                    copy_btn  = gr.Button(
                        "📋 Copy Chat", variant="secondary", scale=1,
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear Chat", variant="secondary", scale=1,
                    )

                audio_out = gr.HTML(value="")

                with gr.Group(visible=False) as chart_group:
                    gr.Markdown("#### Optimisation Results")
                    with gr.Row():
                        chat_fig0 = gr.Plot()
                        chat_fig1 = gr.Plot()
                    with gr.Row():
                        chat_fig2 = gr.Plot()
                        chat_fig3 = gr.Plot()

                tts_source    = gr.Textbox(visible=False, value="", elem_id="tts-source")
                tts_reset_btn = gr.Button(visible=False, elem_id="tts-reset-trigger")

                _chat_outs = [
                    msg_box, chatbot, badges_out, audio_out,
                    chat_fig0, chat_fig1, chat_fig2, chat_fig3, chart_group,
                    tts_source,
                ]
                _SCROLL_JS = """() => {
    setTimeout(function () {
        var w = document.querySelector('#main-chatbot .bubble-wrap');
        if (w) w.scrollTop = w.scrollHeight;
    }, 80);
}"""
                send_btn.click(
                    handle_chat, [msg_box, chatbot, tts_state, portfolio_state], _chat_outs
                ).then(fn=None, js=_SCROLL_JS)
                msg_box.submit(
                    handle_chat, [msg_box, chatbot, tts_state, portfolio_state], _chat_outs
                ).then(fn=None, js=_SCROLL_JS)

                # TTS toggle: JS fires first (synchronous with click → iOS-safe),
                # then Python updates the button label.
                # JS pre-hook: runs sync on click (keeps user-gesture for speech API),
                # passes inputs through unchanged so Python sees the current state.
                _TTS_JS = """(is_on, text) => {
    var synth = window.speechSynthesis;
    if (!synth) return [is_on, text];
    if (synth.speaking || synth.pending) {
        synth.cancel();
        return [is_on, text];
    }
    var src = (text || '').trim();
    if (!src) {
        var el = document.querySelector('#tts-source textarea, #tts-source input');
        if (el) src = el.value.trim();
    }
    if (!src) {
        var bots = document.querySelectorAll('.message.bot');
        if (bots.length) {
            var clone = bots[bots.length-1].cloneNode(true);
            clone.querySelectorAll('button').forEach(function(b) { b.remove(); });
            src = (clone.innerText || '').trim();
        }
    }
    if (!src) return [is_on, text];
    var u = new SpeechSynthesisUtterance(src.slice(0, 2000));
    u.lang = 'en-US';
    u.rate = 1.0;
    u.onend = function() {
        var rb = document.querySelector('#tts-reset-trigger button');
        if (rb) rb.click();
    };
    u.onerror = function() {
        var rb = document.querySelector('#tts-reset-trigger button');
        if (rb) rb.click();
    };
    synth.speak(u);
    return [is_on, text];
}"""

                def _toggle_tts_btn(is_on: bool, _text: str = ""):
                    if is_on:
                        return False, gr.update(value="🔊 Read", elem_classes="tts-btn-off")
                    return True, gr.update(value="⏹ Stop", elem_classes="tts-btn-on")

                def _reset_tts():
                    return False, gr.update(value="🔊 Read", elem_classes="tts-btn-off")

                tts_btn.click(
                    fn=_toggle_tts_btn,
                    inputs=[tts_state, tts_source],
                    outputs=[tts_state, tts_btn],
                    js=_TTS_JS,
                )
                tts_reset_btn.click(fn=_reset_tts, outputs=[tts_state, tts_btn])

                _COPY_JS = """() => {
    const root = document.querySelector('#main-chatbot');
    if (!root) return;
    const text = (root.innerText || '').trim();
    if (!text) return;
    navigator.clipboard.writeText(text);
}"""
                copy_btn.click(fn=None, js=_COPY_JS)

                clear_btn.click(
                    lambda: ([], "", "",
                             _EMPTY, _EMPTY, _EMPTY, _EMPTY, gr.update(visible=False)),
                    outputs=[chatbot, badges_out, audio_out,
                             chat_fig0, chat_fig1, chat_fig2, chat_fig3, chart_group],
                )

                # Sample question buttons → auto-submit directly to the agent
                def _make_sample_fn(q):
                    def _handler(hist, tts, pid):
                        yield from handle_chat(q, hist, tts, pid)
                    return _handler

                for _btn, _q in _sample_btns:
                    _btn.click(
                        fn=_make_sample_fn(_q),
                        inputs=[chatbot, tts_state, portfolio_state],
                        outputs=_chat_outs,
                    ).then(fn=None, js=_SCROLL_JS)

        gr.HTML("""
        <div style="text-align:center;padding:12px 0 6px;color:#374151;font-size:.78rem;">
          Prices via yfinance · KB via Wikipedia/ChromaDB · Optimizer via Markowitz/scipy ·
          Agents via LangGraph
        </div>
        """)

        # ── Portfolio selector — wire events after all components defined ─────
        _switch_outs = (
            [portfolio_state] + _dash_outs + [watchlist_df, remove_dd, edit_dd, prices_label]
            + [chatbot]
            + [opt_status, m_ret, m_vol, m_shrp,
               m_sortino, m_var, m_cash,
               fig_pie, fig_bar, fig_frontier, opt_commentary]
            + [opt_budget, opt_target_vol, opt_rf_slider, opt_rf_text,
               opt_lookback, opt_frontier, opt_sr_slider, opt_sr_text]
        )

        # Fire resize at 150 ms and 500 ms so Plotly re-measures after figures swap,
        # regardless of whether the Dashboard tab is currently visible.
        _RESIZE_JS = ("() => { "
                      "[150, 500].forEach(function(d) { "
                      "  setTimeout(function() { "
                      "    window.dispatchEvent(new Event('resize')); "
                      "  }, d); "
                      "}); }")
        pf_drop.change(_switch_portfolio, [pf_drop], _switch_outs).then(
            fn=None, js=_RESIZE_JS
        )
        # When user navigates to the Dashboard tab, re-measure plots that
        # were rendered while the tab was hidden (display:none).
        dash_tab.select(fn=None, js=_RESIZE_JS)

        # Populate Optimizer tab graphs at app startup (no portfolio re-select needed)
        demo.load(
            _load_saved_optimizer,
            [portfolio_state],
            [opt_status, m_ret, m_vol, m_shrp,
             m_sortino, m_var, m_cash,
             fig_pie, fig_bar, fig_frontier, opt_commentary,
             opt_budget, opt_target_vol, opt_rf_slider, opt_rf_text,
             opt_lookback, opt_frontier, opt_sr_slider, opt_sr_text],
        )

        pf_create_btn.click(
            create_portfolio, [pf_new_name],
            [pf_drop, portfolio_state, pf_status],
        )
        pf_rename_btn.click(
            rename_portfolio, [portfolio_state, pf_rename_name],
            [pf_drop, pf_status],
        )
        pf_delete_btn.click(
            delete_portfolio, [portfolio_state],
            [pf_drop, portfolio_state, pf_status],
        )

    return demo
