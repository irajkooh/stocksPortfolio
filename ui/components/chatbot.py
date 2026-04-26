"""Chatbot component — delegates to the LangGraph multi-agent graph."""
import base64
import io
import logging
from agents.graph import get_graph
from agents.state import empty_state

logger = logging.getLogger(__name__)

_AGENT_LABELS = {
    "market_intel":      "📈 Market Intel",
    "portfolio_analyst": "💼 Portfolio Analyst",
    "risk_manager":      "🛡️ Risk Manager",
    "optimizer":         "🤖 Optimizer",
    "knowledge_base":    "📚 Knowledge Base",
}


def run_agents(
    message: str,
    history: list[list[str | None]],
    portfolio_id: int = 1,
) -> tuple[str, list, list[str], list[str]]:
    """
    Invoke the LangGraph graph for the given portfolio.

    Returns
    -------
    response    : synthesised text answer
    charts      : list of plotly Figure objects (may be empty)
    agents_used : agent keys that were active
    status_log  : human-readable progress lines
    """
    if not message.strip():
        return "", [], [], []

    graph = get_graph()
    state = empty_state(message, portfolio_id=portfolio_id)

    try:
        result      = graph.invoke(state)
        response    = result.get("final_response") or "I couldn't process that."
        charts      = result.get("charts", [])
        agents_used = result.get("active_agents", [])
        status_log  = result.get("agent_status", [])
        return response, charts, agents_used, status_log
    except Exception as exc:
        logger.error("Agent graph error: %s", exc)
        return f"⚠️ Agent error: {exc}", [], [], []


def tts_html(text, enabled: bool = True) -> str:
    """Synthesise *text* with gTTS, return an HTML <audio autoplay> tag.

    Cross-platform (works on HF Space Linux, not just macOS). Returns "" when
    disabled, empty, or on any failure — caller can drop straight into a
    gr.HTML output.
    """
    if not enabled or not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    snippet = text[:2000].replace("\n", " ").strip()
    if not snippet:
        return ""

    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(snippet, lang="en").write_to_fp(buf)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        logger.warning("TTS synthesis failed: %s", exc)
        return ""

    return (
        f'<audio autoplay controls style="width:100%">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
        f'</audio>'
    )


def agent_badges_html(agents_used: list[str]) -> str:
    if not agents_used:
        return ""
    badges = " ".join(
        f'<span class="agent-badge">{_AGENT_LABELS.get(a, a)}</span>'
        for a in agents_used
    )
    return f'<div class="agents-row">Agents: {badges}</div>'
