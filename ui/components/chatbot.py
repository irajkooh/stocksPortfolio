"""Chatbot component — delegates to the LangGraph multi-agent graph."""
import logging
import tempfile
from agents.graph import get_graph
from agents.state import empty_state

logger = logging.getLogger(__name__)

_AGENT_LABELS = {
    "market_intel":      "📈 Market Intel",
    "portfolio_analyst": "💼 Portfolio Analyst",
    "risk_manager":      "🛡️ Risk Manager",
    "rl_optimizer":      "🤖 RL Optimizer",
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


def tts_speak(text: str, enabled: bool) -> str | None:
    if not enabled or not text:
        return None
    try:
        from gtts import gTTS
        tts = gTTS(text=text[:600], lang="en", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            return f.name
    except Exception as exc:
        logger.warning("TTS failed: %s", exc)
        return None


def agent_badges_html(agents_used: list[str]) -> str:
    if not agents_used:
        return ""
    badges = " ".join(
        f'<span class="agent-badge">{_AGENT_LABELS.get(a, a)}</span>'
        for a in agents_used
    )
    return f'<div class="agents-row">Agents: {badges}</div>'
