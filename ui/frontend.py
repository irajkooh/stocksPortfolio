"""Main Gradio Blocks layout — multi-agent powered, multi-portfolio."""
import logging
import os
import re
from datetime import date, timedelta
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from pydantic import ValidationError

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

from core.database import SessionLocal, init_db
from core.models import HoldingCreate, HoldingDB, PortfolioDB
from services.stock_service import (
    get_batch_prices,
    get_stock_info,
    get_period_changes,
    get_historical,
    validate_ticker,
)
from services.llm_service import llm_display_name
from ui.theme import get_theme, CUSTOM_CSS
from ui.components.dashboard import (
    live_watchlist_rows,
    last_plan_rows,
    last_plan_pie,
)
from ui.components.chatbot import run_agents, tts_speak, tts_stop, agent_badges_html
from ui.components.optimizer_ui import (
    run_optimize,
    sync_slider_to_text,
    sync_text_to_slider,
)
from core import runtime
from core import config as _cfg


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
    name = (name or "").strip()
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
            new_p = PortfolioDB(name="Default")
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
    new_name = (new_name or "").strip()
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
    """Return rows for the watchlist dataframe; CASH pinned at top."""
    with SessionLocal() as s:
        tickers = [h.ticker for h in
                   s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()]

    rows: list[list] = [["CASH", "$1.00", "+0.00%", "+0.00%", "+0.00%", "+0.00%"]]
    for t in sorted(set(tickers)):
        try:
            info    = get_stock_info(t) or {}
            periods = get_period_changes(t) or {}
            price   = float(info.get("price") or 0.0)
            rows.append([
                t,
                f"${price:.2f}",
                f"{periods.get('change_1d_pct',  0.0):+.2f}%",
                f"{periods.get('change_1mo_pct', 0.0):+.2f}%",
                f"{periods.get('change_3mo_pct', 0.0):+.2f}%",
                f"{periods.get('change_1y_pct',  0.0):+.2f}%",
            ])
        except Exception as e:
            log.warning("watchlist price lookup failed for %s: %s", t, e)
            rows.append([t, "—", "—", "—", "—", "—"])
    return rows


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


def _portfolio_vs_spy_fig(portfolio_id: int, period: str = "1y") -> go.Figure:
    """Line chart: equal-weight & optimized-weight portfolio cumulative return vs ^GSPC."""
    import json
    from core.models import PortfolioAllocationDB

    with SessionLocal() as s:
        tickers = [h.ticker for h in
                   s.query(HoldingDB).filter_by(portfolio_id=portfolio_id).all()]
        alloc_row = s.get(PortfolioAllocationDB, portfolio_id)

    # Parse optimized weights (stock-only, normalized so they sum to 1)
    opt_weights: dict[str, float] = {}
    if alloc_row:
        allocs = json.loads(alloc_row.allocations_json)
        total_w = sum(v["weight"] for v in allocs.values())
        if total_w > 0:
            opt_weights = {t: v["weight"] / total_w for t, v in allocs.items()}

    fig = go.Figure()
    fig.update_layout(
        title=f"Portfolio vs S&P 500 ({_period_date_range(period)})",
        template="plotly_dark",
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#111827",
        xaxis_title="Date",
        yaxis_title="Normalized (start=100)",
        legend=dict(orientation="h", y=-0.2),
        height=460,
        autosize=True,
        margin=dict(l=48, r=24, t=48, b=60),
    )

    closes: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            hist = get_historical(t, period=period)
            if hist is not None and not hist.empty:
                closes[t] = hist["Close"].dropna()
        except Exception as e:
            log.warning("vs-SPY: history fetch failed for %s: %s", t, e)

    spy = pd.Series(dtype=float)
    if tickers:  # skip network call when portfolio is empty
        spy_hist = get_historical("^GSPC", period=period)
        if spy_hist is not None and not spy_hist.empty:
            spy = spy_hist["Close"].dropna()

    if closes:
        df = pd.concat(closes, axis=1).dropna(how="any")
        if not df.empty:
            norm = df.div(df.iloc[0]).mul(100.0)

            # Equal-weighted line
            port_eq = norm.mean(axis=1)
            fig.add_trace(go.Scatter(
                x=port_eq.index, y=port_eq.values,
                mode="lines", name="Portfolio (equal-weighted)",
                line=dict(color="#00D4FF", width=2),
            ))

            # Optimized-weighted line (only when a saved plan exists)
            if opt_weights:
                common = [t for t in df.columns if t in opt_weights]
                if common:
                    w = pd.Series({t: opt_weights[t] for t in common})
                    w = w / w.sum()
                    opt_port = norm[common].mul(w, axis=1).sum(axis=1)
                    fig.add_trace(go.Scatter(
                        x=opt_port.index, y=opt_port.values,
                        mode="lines", name="Portfolio (optimized)",
                        line=dict(color="#00FF94", width=2, dash="dash"),
                    ))

    if not spy.empty:
        spy_norm = spy.div(spy.iloc[0]).mul(100.0)
        fig.add_trace(go.Scatter(
            x=spy_norm.index, y=spy_norm.values,
            mode="lines", name="S&P 500",
            line=dict(color="#FFA500", width=2, dash="dot"),
        ))

    if not fig.data:
        fig.add_annotation(text="No price history available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#9CA3AF"))
    return fig


def refresh_dashboard(portfolio_id: int = 1):
    watch = live_watchlist_rows(portfolio_id)
    vs_spy = _portfolio_vs_spy_fig(portfolio_id)
    rows, m = last_plan_rows(portfolio_id)
    if m is None:
        return (watch, "—", "—", "—", "—", "—", _placeholder(420), [],
                "_Run the Optimizer to see your plan._", vs_spy)
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


# ── Positions helpers ─────────────────────────────────────────────────────────

# ── Portfolio CRUD ─────────────────────────────────────────────────────────────

def add_ticker(ticker: str, portfolio_id: int):
    try:
        payload = HoldingCreate(ticker=ticker)
    except ValidationError as ve:
        msg = ve.errors()[0].get("msg", "invalid ticker")
        return gr.update(), f"❌ {msg}", gr.update(), gr.update(), _watchlist_df(portfolio_id)
    except Exception as e:
        return gr.update(), f"❌ {e}", gr.update(), gr.update(), _watchlist_df(portfolio_id)
    with SessionLocal() as s:
        exists = (s.query(HoldingDB)
                   .filter_by(portfolio_id=portfolio_id, ticker=payload.ticker)
                   .first())
        if exists:
            return (gr.update(value=""),
                    f"ℹ️ {payload.ticker} already on watchlist",
                    gr.update(), gr.update(),
                    _watchlist_df(portfolio_id))
        s.add(HoldingDB(portfolio_id=portfolio_id, ticker=payload.ticker))
        s.commit()
    rows = _watchlist_df(portfolio_id)
    tickers_for_dropdown = [r[0] for r in rows if r[0] != "CASH"]
    dd = gr.update(choices=tickers_for_dropdown, value=None)
    return gr.update(value=""), f"✅ Added {payload.ticker}", dd, dd, rows


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
        return "⚠️ pick a ticker (CASH is not removable)", rows, gr.update(), gr.update()
    with SessionLocal() as s:
        row = (s.query(HoldingDB)
                .filter_by(portfolio_id=portfolio_id, ticker=ticker).first())
        if row:
            s.delete(row)
            s.commit()
    rows = _watchlist_df(portfolio_id)
    tickers_for_dropdown = [r[0] for r in rows if r[0] != "CASH"]
    dd = gr.update(choices=tickers_for_dropdown, value=None)
    return f"🗑️ Removed {ticker}", rows, dd, dd


# ── Chat handler ───────────────────────────────────────────────────────────────

def handle_chat(message, history, tts_on, portfolio_id: int = 1):
    if not message.strip():
        yield (gr.update(value="", interactive=True), history, "", "",
               _EMPTY, _EMPTY, _EMPTY, _EMPTY, gr.update(visible=False))
        return

    pending = history + [[message, "⏳ Agents working…"]]
    yield (gr.update(value="⏳ Thinking…", interactive=False),
           pending, "", "", _EMPTY, _EMPTY, _EMPTY, _EMPTY, gr.update(visible=False))

    response, charts, agents_used, status_log = run_agents(message, history, portfolio_id)
    new_history = history + [[message, response]]
    badges_html = agent_badges_html(agents_used)
    tts_speak(response, tts_on)

    def _fig(i):
        return charts[i] if i < len(charts) else _EMPTY

    yield (
        gr.update(value="", interactive=True),
        new_history, badges_html, "",
        _fig(0), _fig(1), _fig(2), _fig(3),
        gr.update(visible=bool(charts)),
    )


# ── Portfolio-switch handler (refreshes all tabs) ─────────────────────────────

def _load_saved_optimizer(pid: int) -> tuple:
    """Return 9-element optimizer output tuple from saved PortfolioAllocationDB, or blanks."""
    import json
    from core.database import SessionLocal
    from core.models import PortfolioAllocationDB
    from services.optimizer import build_plots

    with SessionLocal() as s:
        row = s.get(PortfolioAllocationDB, pid)
        if row is None:
            return ("", "", "", "", "", None, None, None, "")
        allocs_json   = row.allocations_json
        frontier_json = row.frontier_json or "[]"
        cash_dollars  = row.cash_dollars
        commentary    = row.commentary or ""
        metrics = {
            "expected_return": row.expected_return,
            "expected_vol":    row.expected_vol,
            "sharpe":          row.sharpe,
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
    return (
        "✅ Last saved",
        f"{metrics['expected_return']*100:.2f}%",
        f"{metrics['expected_vol']*100:.2f}%",
        f"{metrics['sharpe']:.3f}",
        f"${cash_dollars:,.0f}",
        fig_p, fig_b, fig_f,
        commentary,
    )


def _switch_portfolio(choice: str):
    pid  = _id_from_choice(choice)
    dash = refresh_dashboard(pid)
    rows = _watchlist_df(pid)
    tickers = [r[0] for r in rows if r[0] != "CASH"]
    dd = gr.update(choices=tickers, value=None)
    opt = _load_saved_optimizer(pid)
    return (
        pid, *dash, rows, dd, dd,
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
            LangGraph Multi-Agent &nbsp;·&nbsp; RL-Optimised &nbsp;·&nbsp;
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
                gr.Markdown(_date_range_label("Live watchlist"))
                dash_watch = gr.Dataframe(
                    headers=["Ticker", "Price", "1d %", "1mo %", "3mo %", "1y %", "Sharpe (1y)", "Sortino (1y)"],
                    datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    elem_classes=["watchlist-df"],
                )
                gr.Markdown("### Last optimized plan")
                with gr.Row():
                    d_budget = gr.Textbox(label="Budget",          interactive=False)
                    d_ret    = gr.Textbox(label="Expected return",  interactive=False)
                    d_vol    = gr.Textbox(label="Expected vol",     interactive=False)
                    d_shrp   = gr.Textbox(label="Sharpe",           interactive=False)
                    d_cash   = gr.Textbox(label="Cash",             interactive=False)
                dash_pie = gr.Plot(label="Allocation", min_width=400, value=_placeholder(420))
                dash_table = gr.Dataframe(
                    headers=["Ticker", "Weight", "Dollars", "Shares", "Price"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    elem_classes=["watchlist-df"],
                )
                d_stamp = gr.Markdown()
                gr.Markdown(_date_range_label("Portfolio vs S&P 500"))
                dash_vs_spy = gr.Plot(label="Portfolio vs S&P 500", min_width=400, value=_placeholder(460))
                refresh_btn = gr.Button("🔄 Refresh dashboard", variant="primary",
                                        elem_classes="btn-glow")

                _dash_outs = [dash_watch, d_budget, d_ret, d_vol, d_shrp, d_cash,
                              dash_pie, dash_table, d_stamp, dash_vs_spy]
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

                gr.Markdown(_date_range_label("Live prices"))
                watchlist_df = gr.Dataframe(
                    headers=["Ticker", "Price", "1d %", "1mo %", "3mo %", "1y %"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    elem_classes=["watchlist-df"],
                    label="",
                )

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
                    tickers_for_dropdown = [r[0] for r in rows if r[0] != "CASH"]
                    dd = gr.update(choices=tickers_for_dropdown, value=None)
                    return rows, dd

                demo.load(
                    _init_watchlist,
                    [portfolio_state],
                    [watchlist_df, remove_dd],
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
                            m_ret  = gr.Textbox(label="Expected return", interactive=False)
                            m_vol  = gr.Textbox(label="Expected vol",    interactive=False)
                            m_shrp = gr.Textbox(label="Sharpe",          interactive=False)
                            m_cash = gr.Textbox(label="Cash reserve ($)", interactive=False)
                        fig_pie      = gr.Plot(label="Allocation")
                        fig_bar      = gr.Plot(label="Dollar allocation")
                        fig_frontier = gr.Plot(label="Efficient frontier")
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
                opt_btn.click(
                    run_optimize,
                    inputs=[opt_budget, opt_target_vol, opt_rf_text,
                            opt_lookback, opt_frontier, portfolio_state],
                    outputs=[opt_status, m_ret, m_vol, m_shrp, m_cash,
                             fig_pie, fig_bar, fig_frontier, opt_commentary]
                            + _dash_outs,
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
                status_log_out = gr.HTML(value="", visible=False, elem_classes="agent-log")

                # ── Sample questions (3 rows × 4) ─────────────────────────────
                _SAMPLE_QS = [
                    "How you can help me?",
                    "what are my holdings?",
                    "What's my total P&L today?",
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
                        elem_classes="tts-btn-off",
                    )
                    copy_btn  = gr.Button(
                        "📋 Copy Chat", variant="secondary", scale=1,
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear Chat", variant="secondary", scale=1,
                    )

                audio_out = gr.HTML(value="")

                with gr.Group(visible=False) as chart_group:
                    gr.Markdown("#### RL Optimisation Results")
                    with gr.Row():
                        chat_fig0 = gr.Plot()
                        chat_fig1 = gr.Plot()
                    with gr.Row():
                        chat_fig2 = gr.Plot()
                        chat_fig3 = gr.Plot()

                _chat_outs = [
                    msg_box, chatbot, badges_out, audio_out,
                    chat_fig0, chat_fig1, chat_fig2, chat_fig3, chart_group,
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

                # TTS toggle: speak last assistant message, or stop playback in progress.
                def _toggle_tts(is_on: bool, history: list):
                    if is_on:
                        tts_stop()
                        return (
                            False,
                            gr.update(value="🔊 Read", variant="secondary",
                                      elem_classes="tts-btn-off"),
                        )
                    def _as_text(x) -> str:
                        if x is None:
                            return ""
                        if isinstance(x, str):
                            return x
                        if isinstance(x, dict):
                            return _as_text(x.get("text") or x.get("content") or "")
                        if isinstance(x, (list, tuple)):
                            return " ".join(_as_text(i) for i in x)
                        return str(x)

                    last = ""
                    for msg in reversed(history or []):
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            last = _as_text(msg.get("content"))
                            if last:
                                break
                        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                            last = _as_text(msg[1])
                            if last:
                                break
                    if not last.strip():
                        return (False, gr.update())
                    tts_speak(last, enabled=True)
                    return (
                        True,
                        gr.update(value="⏹ Stop", variant="stop",
                                  elem_classes="tts-btn-on"),
                    )

                tts_btn.click(_toggle_tts, [tts_state, chatbot], [tts_state, tts_btn])

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
            [portfolio_state] + _dash_outs + [watchlist_df, remove_dd, edit_dd]
            + [chatbot]
            + [opt_status, m_ret, m_vol, m_shrp, m_cash,
               fig_pie, fig_bar, fig_frontier, opt_commentary]
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
            [opt_status, m_ret, m_vol, m_shrp, m_cash,
             fig_pie, fig_bar, fig_frontier, opt_commentary],
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
