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
        block_label_text_color        = "#D1D5DB",
        input_background_fill         = PALETTE["surface2"],
        input_background_fill_dark    = PALETTE["surface2"],
        input_border_color            = PALETTE["border"],
        input_border_color_focus      = PALETTE["cyan"],
        button_primary_background_fill       = f"linear-gradient(135deg, {PALETTE['cyan']}, {PALETTE['purple']})",
        button_primary_background_fill_hover = "linear-gradient(135deg, #00B8E0, #6A1FA6)",
        button_primary_text_color            = "white",
        button_secondary_background_fill      = PALETTE["surface2"],
        button_secondary_background_fill_hover= PALETTE["border"],
        button_secondary_text_color          = "white",
        button_cancel_background_fill        = "#7F1D1D",
        button_cancel_background_fill_hover  = "#991B1B",
        button_cancel_text_color             = "white",
        # ── Text visibility ──────────────────────────────────────
        body_text_color                      = "#E5E7EB",
        body_text_color_subdued              = "#9CA3AF",
        block_title_text_color               = "#E5E7EB",
        block_title_text_color_dark          = "#E5E7EB",
        block_label_text_color_dark          = "#D1D5DB",
        checkbox_label_text_color            = "#D1D5DB",
        input_placeholder_color              = "#6B7280",
        accordion_text_color                 = "#D1D5DB",
        accordion_text_color_dark            = "#D1D5DB",
        table_text_color                     = "#E5E7EB",
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

/* ── Chatbot response headings — keep small, never H1/H2 size ── */
.message.bot h1, .message.bot h2, .message.bot h3,
.message.bot h4, .message.bot h5, .message.bot h6 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    margin: 0.6em 0 0.2em !important;
    color: #93C5FD !important;
}

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

/* ── Global text visibility ──────────────────────────────── */
body, .gradio-container, .gradio-container * {
    color: #E5E7EB;
}
label, .label-wrap, .label-wrap span, .block-label span {
    color: #D1D5DB !important;
}
input, textarea, select {
    color: #E5E7EB !important;
    caret-color: #00D4FF !important;
}
input::placeholder, textarea::placeholder {
    color: #6B7280 !important;
    opacity: 1 !important;
}
select option {
    background: #1F2937 !important;
    color: #E5E7EB !important;
}

/* ── Dropdown / listbox options ─────────────────────────── */
ul.options, ul.options li, .dropdown-arrow,
[data-testid="dropdown"] li,
.multiselect-dropdown li {
    background: #1F2937 !important;
    color: #E5E7EB !important;
}
ul.options li:hover, [data-testid="dropdown"] li:hover {
    background: #374151 !important;
}

/* ── Dataframe cells ─────────────────────────────────────── */
.dataframe tbody td {
    color: #E5E7EB !important;
    background: #111827 !important;
}
.dataframe tbody tr:nth-child(even) td {
    background: #0d1118 !important;
}

/* ── Chatbot message text ────────────────────────────────── */
.message p, .message div, .message span,
.message.user p, .message.bot p,
.bubble-wrap .message { color: #E5E7EB !important; }

/* ── Accordion / collapse headers ───────────────────────── */
.accordion-header, details summary { color: #D1D5DB !important; }

/* ── Slider labels & number inputs ──────────────────────── */
input[type="number"], input[type="range"] + span {
    color: #E5E7EB !important;
}

/* ── Radio & checkbox labels ─────────────────────────────── */
.radio-group label, .checkbox-group label,
fieldset label, .wrap label {
    color: #D1D5DB !important;
}

/* ── Block/panel titles ──────────────────────────────────── */
.block-label, .block-label span,
.svelte-1gfkn6j, .label-wrap { color: #D1D5DB !important; }

/* ── Sample question buttons ─────────────────────────────── */
button.sample-q, .sample-q button {
    font-size: .72rem !important;
    padding: 4px 8px !important;
    white-space: normal !important;
    text-align: left !important;
    line-height: 1.3 !important;
    min-height: 40px !important;
    color: #9CA3AF !important;
    border-color: #2D3748 !important;
}
button.sample-q:hover, .sample-q button:hover {
    color: #00D4FF !important;
    border-color: #00D4FF !important;
    background: #0d1929 !important;
}

/* ── TTS toggle button states ────────────────────────────── */
.tts-btn-off button, button.tts-btn-off {
    color: #00FF94 !important;
    border-color: #00FF94 !important;
    background: rgba(0,255,148,.08) !important;
}
.tts-btn-off button:hover, button.tts-btn-off:hover {
    background: rgba(0,255,148,.18) !important;
    box-shadow: 0 0 12px rgba(0,255,148,.35) !important;
}
.tts-btn-on button, button.tts-btn-on {
    color: #FF4757 !important;
    border-color: #FF4757 !important;
    background: rgba(255,71,87,.12) !important;
}
.tts-btn-on button:hover, button.tts-btn-on:hover {
    background: rgba(255,71,87,.22) !important;
    box-shadow: 0 0 12px rgba(255,71,87,.4) !important;
}

/* ── Runtime banner (top of UI) ───────────────────────── */
.runtime-banner {
    color: #FFD700 !important;
    background: #0d1118 !important;
    border: 1px solid #2a2a1a !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    margin: 0 0 10px !important;
    font-family: ui-monospace, "JetBrains Mono", monospace !important;
    font-size: .82rem !important;
    letter-spacing: .02em !important;
}

/* ── Chat input: yellow italic text while thinking ───── */
#chat-input textarea:disabled {
    color: #EAB308 !important;
    font-weight: 600 !important;
    font-style: italic !important;
    opacity: 1 !important;
}

/* ── Plot containers: always fill width ──────────────── */
.js-plotly-plot, .plot-container, .plot-container.plotly,
.svelte-1cl284s, [data-testid="plot"] {
    width: 100% !important;
}
.js-plotly-plot .plotly { width: 100% !important; }

/* ── Watchlist dataframe: scrollable on mobile ───────── */
.watchlist-df table { table-layout: auto !important; }
.watchlist-df thead,
.watchlist-df tbody { width: auto !important; }
.watchlist-df th,
.watchlist-df td { white-space: nowrap !important; }

/* ── Watchlist dataframe: black on white ─────────────── */
.watchlist-df,
.watchlist-df *,
.watchlist-df table,
.watchlist-df thead,
.watchlist-df tbody,
.watchlist-df tr,
.watchlist-df th,
.watchlist-df td,
.watchlist-df span,
.watchlist-df div {
    background-color: #fff !important;
    color: #000 !important;
}

"""
