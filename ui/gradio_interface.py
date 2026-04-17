"""Main Gradio Blocks layout — multi-agent powered, multi-portfolio."""
import re
import gradio as gr
import pandas as pd
import plotly.graph_objects as go

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
    S -->|rebalance / optimise| RL["🤖 RL Optimizer Agent\nPPO · stable-baselines3\nBudget → $ allocations\nAUTO-TRIGGERED"]
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
from core.models import HoldingDB, Portfolio
from services.stock_service import get_batch_prices, get_stock_info, validate_ticker
from services.llm_service import llm_display_name
from ui.theme import get_theme, CUSTOM_CSS
from ui.components.dashboard import allocation_pie, pnl_bar, performance_chart, sector_bar
from ui.components.chatbot import run_agents, tts_speak, agent_badges_html
from ui.components.optimizer_ui import run_and_render

_EMPTY = go.Figure()
_EMPTY.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
                     font=dict(color="white"))


# ── Portfolio management helpers ──────────────────────────────────────────────

def _list_portfolios() -> list:
    db = SessionLocal()
    try:
        return db.query(Portfolio).order_by(Portfolio.name).all()
    finally:
        db.close()


def _portfolio_choices() -> list[str]:
    return [f"{p.name}  (#{p.id})" for p in _list_portfolios()]


def _id_from_choice(choice: str | None) -> int:
    if not choice:
        return 1
    m = re.search(r'#(\d+)', choice)
    return int(m.group(1)) if m else 1


def create_portfolio(name: str):
    name = (name or "").strip()
    if not name:
        return gr.update(), gr.update(), "❌ Enter a portfolio name."
    db = SessionLocal()
    try:
        if db.query(Portfolio).filter(Portfolio.name == name).first():
            return gr.update(), gr.update(), f"❌ **{name}** already exists."
        p = Portfolio(name=name)
        db.add(p)
        db.commit()
        db.refresh(p)
        choices = _portfolio_choices()
        new_choice = f"{p.name}  (#{p.id})"
        return (
            gr.update(choices=choices, value=new_choice),
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
        p = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not p:
            return gr.update(), portfolio_id, "❌ Portfolio not found."
        name = p.name
        db.delete(p)
        db.commit()

        choices = _portfolio_choices()
        if not choices:
            new_p = Portfolio(name="Default")
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
        p = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not p:
            return gr.update(), "❌ Portfolio not found."
        p.name = new_name
        db.commit()
        db.refresh(p)
        choices = _portfolio_choices()
        current_choice = f"{p.name}  (#{p.id})"
        return (
            gr.update(choices=choices, value=current_choice),
            f"✅ Renamed to **{new_name}**.",
        )
    except Exception as e:
        return gr.update(), f"❌ {e}"
    finally:
        db.close()


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_summary(portfolio_id: int = 1) -> dict:
    db = SessionLocal()
    try:
        holdings = (
            db.query(HoldingDB)
            .filter(HoldingDB.portfolio_id == portfolio_id)
            .all()
        )
        if not holdings:
            return {"total_value": 0, "total_cost": 0,
                    "total_pnl": 0, "total_pnl_pct": 0, "holdings": []}
        prices = get_batch_prices([h.ticker for h in holdings])
        rows, tv, tc = [], 0.0, 0.0
        for h in holdings:
            price = prices.get(h.ticker, 0.0)
            value = price * h.shares
            cost  = h.purchase_price * h.shares
            pnl   = value - cost
            tv += value
            tc += cost
            info = get_stock_info(h.ticker)
            rows.append({
                "ticker":         h.ticker,
                "name":           info.get("name", h.ticker),
                "shares":         h.shares,
                "purchase_price": h.purchase_price,
                "current_price":  price,
                "value":          round(value, 2),
                "cost":           round(cost, 2),
                "pnl":            round(pnl, 2),
                "pnl_pct":        round(pnl / cost * 100, 2) if cost else 0.0,
                "sector":         info.get("sector", "Unknown"),
                "change_pct":     info.get("change_pct", 0.0),
            })
        pnl_t = tv - tc
        return {
            "total_value":   round(tv, 2),
            "total_cost":    round(tc, 2),
            "total_pnl":     round(pnl_t, 2),
            "total_pnl_pct": round(pnl_t / tc * 100, 2) if tc else 0.0,
            "holdings":      rows,
        }
    finally:
        db.close()


def _holdings_df(portfolio_id: int = 1) -> pd.DataFrame:
    holdings = _load_summary(portfolio_id)["holdings"]
    if not holdings:
        return pd.DataFrame(columns=[
            "Ticker", "Name", "Shares", "Buy $",
            "Current $", "Value", "P&L", "P&L %", "Today %", "Sector",
        ])
    return pd.DataFrame([{
        "Ticker":    h["ticker"],
        "Name":      h["name"][:22],
        "Shares":    f"{h['shares']:.4f}",
        "Buy $":     f"${h['purchase_price']:.2f}",
        "Current $": f"${h['current_price']:.2f}",
        "Value":     f"${h['value']:,.2f}",
        "P&L":       f"${h['pnl']:+,.2f}",
        "P&L %":     f"{h['pnl_pct']:+.2f}%",
        "Today %":   f"{h['change_pct']:+.2f}%",
        "Sector":    h["sector"],
    } for h in holdings])


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

def refresh_dashboard(portfolio_id: int = 1):
    s = _load_summary(portfolio_id)
    h = s["holdings"]
    return (
        f"${s['total_value']:,.2f}",
        f"${s['total_pnl']:+,.2f}",
        f"{s['total_pnl_pct']:+.2f} %",
        f"${s['total_cost']:,.2f}",
        allocation_pie(h),
        pnl_bar(h),
        performance_chart([x["ticker"] for x in h]),
        sector_bar(h),
        _holdings_df(portfolio_id),
    )


# ── Portfolio CRUD ─────────────────────────────────────────────────────────────

def add_holding(ticker, shares, price, portfolio_id: int = 1):
    if not ticker:
        return "**Enter a ticker.**", _holdings_df(portfolio_id)
    ticker = ticker.strip().upper()
    db = SessionLocal()
    try:
        existing = (
            db.query(HoldingDB)
            .filter(HoldingDB.portfolio_id == portfolio_id,
                    HoldingDB.ticker == ticker)
            .first()
        )
        if existing:
            return f"❌ **{ticker}** already in this portfolio.", _holdings_df(portfolio_id)
        if not validate_ticker(ticker):
            return f"❌ **{ticker}** is not a valid ticker.", _holdings_df(portfolio_id)
        db.add(HoldingDB(
            ticker=ticker, shares=shares,
            purchase_price=price, portfolio_id=portfolio_id,
        ))
        db.commit()
        return f"✅ Added **{shares}** × **{ticker}** @ ${price:.2f}", _holdings_df(portfolio_id)
    except Exception as e:
        return f"❌ {e}", _holdings_df(portfolio_id)
    finally:
        db.close()


def remove_holding(ticker, portfolio_id: int = 1):
    if not ticker:
        return "**Enter a ticker.**", _holdings_df(portfolio_id)
    ticker = ticker.strip().upper()
    db = SessionLocal()
    try:
        h = (
            db.query(HoldingDB)
            .filter(HoldingDB.portfolio_id == portfolio_id,
                    HoldingDB.ticker == ticker)
            .first()
        )
        if not h:
            return f"❌ **{ticker}** not found.", _holdings_df(portfolio_id)
        db.delete(h)
        db.commit()
        return f"✅ Removed **{ticker}**.", _holdings_df(portfolio_id)
    except Exception as e:
        return f"❌ {e}", _holdings_df(portfolio_id)
    finally:
        db.close()


def update_holding(ticker, shares, price, portfolio_id: int = 1):
    if not ticker:
        return "**Enter a ticker.**", _holdings_df(portfolio_id)
    ticker = ticker.strip().upper()
    db = SessionLocal()
    try:
        h = (
            db.query(HoldingDB)
            .filter(HoldingDB.portfolio_id == portfolio_id,
                    HoldingDB.ticker == ticker)
            .first()
        )
        if not h:
            return f"❌ **{ticker}** not found.", _holdings_df(portfolio_id)
        if shares > 0:
            h.shares = shares
        if price > 0:
            h.purchase_price = price
        db.commit()
        return f"✅ Updated **{ticker}**.", _holdings_df(portfolio_id)
    except Exception as e:
        return f"❌ {e}", _holdings_df(portfolio_id)
    finally:
        db.close()


# ── Chat handler ───────────────────────────────────────────────────────────────

def handle_chat(message, history, tts_on, portfolio_id: int = 1):
    if not message.strip():
        return ("", history, "", gr.update(visible=False),
                _EMPTY, _EMPTY, _EMPTY, _EMPTY, gr.update(visible=False))

    response, charts, agents_used, status_log = run_agents(message, history, portfolio_id)
    new_history = history + [[message, response]]
    badges_html = agent_badges_html(agents_used)
    audio_path  = tts_speak(response, tts_on)
    audio_upd   = gr.update(value=audio_path, visible=bool(audio_path))

    def _fig(i):
        return charts[i] if i < len(charts) else _EMPTY

    return (
        "", new_history, badges_html, audio_upd,
        _fig(0), _fig(1), _fig(2), _fig(3),
        gr.update(visible=bool(charts)),
    )


# ── Portfolio-switch handler (refreshes all tabs) ─────────────────────────────

def _switch_portfolio(choice: str):
    pid  = _id_from_choice(choice)
    dash = refresh_dashboard(pid)
    df   = _holdings_df(pid)
    tkrs = _portfolio_tickers_str(pid)
    return (pid, *dash, df, tkrs)


# ── Interface factory ─────────────────────────────────────────────────────────

def create_interface() -> gr.Blocks:
    init_db()
    initial_choices = _portfolio_choices()
    initial_choice  = initial_choices[0] if initial_choices else None

    with gr.Blocks(theme=get_theme(), css=CUSTOM_CSS,
                   title="AI Portfolio Manager", analytics_enabled=False) as demo:

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

        with gr.Tabs(elem_classes="tab-nav"):

            # ════════════════════════════════════════════════════════════════
            # TAB 1 — Dashboard
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("📊 Dashboard"):
                with gr.Row():
                    val_box  = gr.Textbox(label="Total Value",  interactive=False, elem_classes="metric-card")
                    pnl_box  = gr.Textbox(label="Total P&L",    interactive=False, elem_classes="metric-card")
                    pct_box  = gr.Textbox(label="P&L %",        interactive=False, elem_classes="metric-card")
                    cost_box = gr.Textbox(label="Invested",     interactive=False, elem_classes="metric-card")

                refresh_btn = gr.Button("🔄 Refresh", variant="primary", elem_classes="btn-glow")

                with gr.Row():
                    alloc_fig  = gr.Plot(label="Allocation")
                    pnl_fig    = gr.Plot(label="P&L")
                perf_fig   = gr.Plot(label="1-Year Performance")
                sector_fig = gr.Plot(label="Sector Exposure")
                dash_table = gr.DataFrame(label="Holdings", interactive=False)

                _dash_outs = [val_box, pnl_box, pct_box, cost_box,
                              alloc_fig, pnl_fig, perf_fig, sector_fig, dash_table]
                refresh_btn.click(refresh_dashboard, [portfolio_state], _dash_outs)
                demo.load(refresh_dashboard, [portfolio_state], _dash_outs)

            # ════════════════════════════════════════════════════════════════
            # TAB 2 — Manage Holdings
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("💼 Portfolio"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ➕ Add")
                        t_add   = gr.Textbox(placeholder="AAPL", label="Ticker")
                        s_add   = gr.Number(value=1.0,   label="Shares",     minimum=0.0001)
                        p_add   = gr.Number(value=100.0, label="Buy Price $", minimum=0.01)
                        add_btn = gr.Button("Add", variant="primary", elem_classes="btn-glow")
                        add_st  = gr.Markdown()

                    with gr.Column():
                        gr.Markdown("### ✏️ Update")
                        t_upd   = gr.Textbox(placeholder="AAPL", label="Ticker")
                        s_upd   = gr.Number(value=0.0, label="New Shares (0=keep)")
                        p_upd   = gr.Number(value=0.0, label="New Price $  (0=keep)")
                        upd_btn = gr.Button("Update", variant="secondary")
                        upd_st  = gr.Markdown()

                    with gr.Column():
                        gr.Markdown("### 🗑️ Remove")
                        t_rm   = gr.Textbox(placeholder="AAPL", label="Ticker")
                        rm_btn = gr.Button("Remove", variant="stop")
                        rm_st  = gr.Markdown()

                gr.Markdown("### Holdings")
                mgmt_table = gr.DataFrame(interactive=False)
                gr.Button("🔄 Reload", size="sm").click(
                    _holdings_df, [portfolio_state], mgmt_table
                )

                add_btn.click(add_holding, [t_add, s_add, p_add, portfolio_state], [add_st, mgmt_table])
                upd_btn.click(update_holding, [t_upd, s_upd, p_upd, portfolio_state], [upd_st, mgmt_table])
                rm_btn.click(remove_holding, [t_rm, portfolio_state], [rm_st, mgmt_table])
                demo.load(_holdings_df, [portfolio_state], mgmt_table)

            # ════════════════════════════════════════════════════════════════
            # TAB 3 — RL Optimizer (standalone, direct)
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("🤖 RL Optimizer"):
                gr.Markdown("""
### Direct RL Portfolio Optimiser
Tickers auto-populate from the selected portfolio, or enter manually.
The AI Assistant tab also triggers this automatically when you say *"rebalance"*.
                """)
                with gr.Row():
                    opt_tickers = gr.Textbox(
                        label="Tickers (comma-separated)",
                        placeholder="AAPL, MSFT, GOOGL, AMZN, TSLA",
                        scale=3,
                    )
                    opt_budget = gr.Number(value=100_000, label="Budget ($)", minimum=1, scale=1)
                    opt_period = gr.Dropdown(["1y", "2y", "3y", "5y"], value="2y", label="History", scale=1)
                    opt_steps  = gr.Slider(
                        2_000, 50_000, value=10_000, step=1_000,
                        label="Training Steps", scale=2,
                    )
                with gr.Row():
                    load_tickers_btn = gr.Button(
                        "📥 Load from Portfolio", size="sm", variant="secondary"
                    )
                    opt_btn = gr.Button(
                        "🚀 Run Optimisation", variant="primary", elem_classes="btn-glow"
                    )
                opt_status = gr.Markdown()
                with gr.Row():
                    fig_ow = gr.Plot(label="Allocation")
                    fig_op = gr.Plot(label="RL vs Equal Weight")
                with gr.Row():
                    fig_of = gr.Plot(label="Efficient Frontier")
                    fig_ob = gr.Plot(label="Dollar Allocation")

                load_tickers_btn.click(_portfolio_tickers_str, [portfolio_state], opt_tickers)
                opt_btn.click(
                    run_and_render,
                    [opt_tickers, opt_budget, opt_period, opt_steps],
                    [opt_status, fig_ow, fig_op, fig_of, fig_ob],
                )
                demo.load(_portfolio_tickers_str, [portfolio_state], opt_tickers)

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
                  Say <em>"rebalance my portfolio with $50k"</em> → RL Optimizer triggers automatically.
                </p>
                """)

                show_workflow = gr.Checkbox(
                    label="🔀 Show Agent Workflow Diagram",
                    value=False, interactive=True,
                )
                workflow_diagram = gr.HTML(value="", visible=False,
                                           elem_classes="mermaid-wrap")
                show_workflow.change(
                    fn=lambda on: gr.update(value=_MERMAID_HTML if on else "", visible=on),
                    inputs=show_workflow,
                    outputs=workflow_diagram,
                )

                chatbot = gr.Chatbot(
                    label="", height=400, bubble_full_width=False,
                    avatar_images=(
                        None,
                        "https://cdn-icons-png.flaticon.com/512/4616/4616734.png",
                    ),
                )

                badges_out     = gr.HTML(value="", visible=True)
                status_log_out = gr.HTML(value="", visible=False, elem_classes="agent-log")

                with gr.Row():
                    msg_box  = gr.Textbox(
                        placeholder=(
                            "Ask anything: 'How is my portfolio?' · "
                            "'Rebalance with $100k' · 'What is Sharpe ratio?'"
                        ),
                        show_label=False, scale=5,
                    )
                    send_btn = gr.Button(
                        "Send 📨", variant="primary", scale=1, elem_classes="btn-glow"
                    )

                with gr.Row():
                    tts_chk  = gr.Checkbox(label="🔊 Read aloud", value=False)
                    stop_btn = gr.Button("⏹ Stop", size="sm", variant="secondary")

                audio_out = gr.Audio(autoplay=True, visible=False,
                                     show_label=False, format="mp3")
                clear_btn = gr.Button("🗑️ Clear", size="sm", variant="secondary")

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
                send_btn.click(
                    handle_chat, [msg_box, chatbot, tts_chk, portfolio_state], _chat_outs
                )
                msg_box.submit(
                    handle_chat, [msg_box, chatbot, tts_chk, portfolio_state], _chat_outs
                )

                stop_btn.click(lambda: gr.update(value=None, visible=False), outputs=audio_out)
                clear_btn.click(
                    lambda: ([], "", gr.update(visible=False),
                             _EMPTY, _EMPTY, _EMPTY, _EMPTY, gr.update(visible=False)),
                    outputs=[chatbot, badges_out, audio_out,
                             chat_fig0, chat_fig1, chat_fig2, chat_fig3, chart_group],
                )

        gr.HTML("""
        <div style="text-align:center;padding:12px 0 6px;color:#374151;font-size:.78rem;">
          Prices via yfinance · KB via Wikipedia/ChromaDB · RL via stable-baselines3 ·
          Agents via LangGraph
        </div>
        """)

        # ── Portfolio selector — wire events after all components defined ─────
        _switch_outs = [portfolio_state] + _dash_outs + [mgmt_table, opt_tickers]

        pf_drop.change(_switch_portfolio, [pf_drop], _switch_outs)

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
