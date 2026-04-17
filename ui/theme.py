import gradio as gr

PALETTE = dict(
    bg        = "#0A0E1A",
    surface   = "#111827",
    surface2  = "#1F2937",
    border    = "#374151",
    cyan      = "#00D4FF",
    purple    = "#7B2FBE",
    green     = "#00FF94",
    red       = "#FF4757",
    amber     = "#FFD700",
    text_muted= "#9CA3AF",
)


def get_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill          = PALETTE["bg"],
        body_background_fill_dark     = PALETTE["bg"],
        block_background_fill         = PALETTE["surface"],
        block_background_fill_dark    = PALETTE["surface"],
        block_border_color            = PALETTE["border"],
        block_border_color_dark       = PALETTE["border"],
        block_label_text_color        = PALETTE["text_muted"],
        input_background_fill         = PALETTE["surface2"],
        input_background_fill_dark    = PALETTE["surface2"],
        input_border_color            = PALETTE["border"],
        input_border_color_focus      = PALETTE["cyan"],
        button_primary_background_fill       = f"linear-gradient(135deg, {PALETTE['cyan']}, {PALETTE['purple']})",
        button_primary_background_fill_hover = f"linear-gradient(135deg, #00B8E0, #6A1FA6)",
        button_primary_text_color            = "white",
        button_secondary_background_fill      = PALETTE["surface2"],
        button_secondary_background_fill_hover= PALETTE["border"],
        button_secondary_text_color          = "white",
        button_cancel_background_fill        = "#7F1D1D",
        button_cancel_background_fill_hover  = "#991B1B",
        button_cancel_text_color             = "white",
    )


CUSTOM_CSS = """
/* ── Body ──────────────────────────────────────────────── */
.gradio-container { background: #0A0E1A !important; }
body { background: #0A0E1A !important; }

/* ── Header gradient text ──────────────────────────────── */
.app-title {
    background: linear-gradient(135deg, #00D4FF, #7B2FBE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem; font-weight: 800; margin: 0;
}

/* ── Portfolio selector bar ─────────────────────────────── */
.portfolio-bar {
    background: linear-gradient(135deg, #111827, #0d1929) !important;
    border: 1px solid #1e3a4a !important;
    border-radius: 14px !important;
    padding: 12px 16px !important;
    margin: 0 0 12px !important;
}
.portfolio-bar label { color: #00D4FF !important; font-weight: 700 !important; }

/* ── Metric cards ───────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #111827, #1F2937) !important;
    border: 1px solid #374151 !important;
    border-radius: 14px !important;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 28px rgba(0,212,255,.18) !important;
}
.metric-card textarea, .metric-card input {
    font-size: 1.4rem !important; font-weight: 700 !important;
    color: #00D4FF !important; text-align: center !important;
}

/* ── Buttons — click bounce ─────────────────────────────── */
button { transition: all .14s ease !important; }
button:active {
    transform: scale(.95) !important;
    box-shadow: 0 0 18px rgba(0,212,255,.55) !important;
}
.btn-glow { box-shadow: 0 0 12px rgba(0,212,255,.3) !important; }

/* ── Tab nav ─────────────────────────────────────────────── */
.tab-nav > div { background: #111827 !important; border-bottom: 1px solid #374151; }
.tab-nav button { color: #9CA3AF !important; font-weight: 600 !important; font-size: .93rem; }
.tab-nav button.selected {
    color: #00D4FF !important;
    border-bottom: 3px solid #00D4FF !important;
    background: transparent !important;
}

/* ── Chatbot bubbles ─────────────────────────────────────── */
.message.user   { background: #1F2937 !important; border: 1px solid #374151; }
.message.bot    { background: #0d1929 !important; border: 1px solid #1F2937; }

/* ── Dataframe ───────────────────────────────────────────── */
.dataframe thead tr th {
    background: #1F2937 !important;
    color: #9CA3AF !important;
    font-size: .75rem; text-transform: uppercase; letter-spacing: .05em;
}
.dataframe tbody tr:hover td { background: #1a2235 !important; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track  { background: #111827; }
::-webkit-scrollbar-thumb  { background: #374151; border-radius: 4px; }

/* ── Positive / negative colours ────────────────────────── */
.pos { color: #00FF94 !important; font-weight: 700; }
.neg { color: #FF4757 !important; font-weight: 700; }

/* ── Status banner ───────────────────────────────────────── */
.status-ok  { color: #00FF94 !important; }
.status-err { color: #FF4757 !important; }

/* ── Agent badges ────────────────────────────────────────── */
.agents-row {
    padding: 6px 0 2px;
    font-size: .78rem;
    color: #9CA3AF;
}
.agent-badge {
    display: inline-block;
    background: linear-gradient(135deg,#1F2937,#374151);
    border: 1px solid #4B5563;
    border-radius: 20px;
    padding: 2px 10px;
    margin: 2px 3px;
    font-size: .75rem;
    color: #00D4FF;
    font-weight: 600;
}

/* ── Agent status log ────────────────────────────────────── */
.agent-log {
    background: #0d1118;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: .78rem;
    color: #6B7280;
    font-family: monospace;
    max-height: 90px;
    overflow-y: auto;
}

/* ── Mermaid workflow diagram ────────────────────────────── */
.mermaid-wrap {
    background: #0d1118;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    overflow-x: auto;
}
.mermaid-wrap svg { max-width: 100%; }
"""
