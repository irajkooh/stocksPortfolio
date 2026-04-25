"""Chatbot component — delegates to the LangGraph multi-agent graph."""
import logging
import subprocess
import sys
import threading
from agents.graph import get_graph
from agents.state import empty_state

logger = logging.getLogger(__name__)

_tts_proc: subprocess.Popen | None = None
_tts_lock = threading.Lock()

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


def tts_stop() -> None:
    """Terminate any in-progress `say` playback."""
    global _tts_proc
    with _tts_lock:
        proc, _tts_proc = _tts_proc, None
    if proc and proc.poll() is None:
        try:
            proc.terminate()
        except Exception as exc:
            logger.warning("TTS stop failed: %s", exc)


def tts_speak(text, enabled: bool = True) -> None:
    """Play TTS through macOS `say` in a background thread. Cancels any prior playback."""
    if not enabled or not text:
        return
    if sys.platform != "darwin":
        logger.warning("TTS: unsupported platform %s", sys.platform)
        return

    if not isinstance(text, str):
        text = str(text)
    snippet = text[:2000].replace("\n", " ").strip()
    if not snippet:
        return

    tts_stop()

    def _play() -> None:
        global _tts_proc
        try:
            proc = subprocess.Popen(["say", snippet])
            with _tts_lock:
                _tts_proc = proc
            proc.wait()
        except Exception as exc:
            logger.warning("TTS playback failed: %s", exc)

    threading.Thread(target=_play, daemon=True).start()


def agent_badges_html(agents_used: list[str]) -> str:
    if not agents_used:
        return ""
    badges = " ".join(
        f'<span class="agent-badge">{_AGENT_LABELS.get(a, a)}</span>'
        for a in agents_used
    )
    return f'<div class="agents-row">Agents: {badges}</div>'
