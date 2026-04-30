"""Chatbot component — delegates to the LangGraph multi-agent graph."""
import base64
import io
import logging
import re
from agents.graph import get_graph
from agents.state import empty_state

logger = logging.getLogger(__name__)

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_BLOCK   = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
_HTML_TAG     = re.compile(r"<[^>]+>")
_HTML_ENTITY  = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
_MD_FENCE    = re.compile(r"```.*?```", re.DOTALL)
_MD_LINK     = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_BOLD     = re.compile(r"(\*\*|__)(.+?)\1")
_MD_ITALIC   = re.compile(r"(?<!\w)([*_])([^*_\n]+?)\1(?!\w)")
_MD_CODE     = re.compile(r"`+([^`]*)`+")
_MD_HEAD     = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_QUOTE    = re.compile(r"^\s*>\s?", re.MULTILINE)
_MD_LIST     = re.compile(r"^\s*([-*+]|\d+\.)\s+", re.MULTILINE)
_MD_HRULE    = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)
_MD_TABLESEP = re.compile(r"^\s*\|?\s*:?-+:?(\s*\|\s*:?-+:?)+\s*\|?\s*$", re.MULTILINE)
_PIPE        = re.compile(r"\|")
_EMOJI       = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F1FF"
    "\U0001F900-\U0001F9FF"
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0000200D"              # zero-width joiner
    "✀-➿"
    "]+",
    flags=re.UNICODE,
)
_LEFTOVER    = re.compile(r"[*_`#>|~]")
_REPEAT_RUN  = re.compile(r"([A-Za-z0-9])\1{3,}")
_MULTI_SPACE = re.compile(r"\s+")


def _strip_for_tts(text: str) -> str:
    """Strip markdown formatting + emojis so TTS reads natural prose only."""
    text = _HTML_COMMENT.sub(" ", text)
    text = _HTML_BLOCK.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)
    text = _MD_FENCE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_BOLD.sub(r"\2", text)
    text = _MD_ITALIC.sub(r"\2", text)
    text = _MD_CODE.sub(r"\1", text)
    text = _MD_HEAD.sub("", text)
    text = _MD_QUOTE.sub("", text)
    text = _MD_LIST.sub("", text)
    text = _MD_HRULE.sub("", text)
    text = _MD_TABLESEP.sub("", text)
    text = _PIPE.sub(" ", text)
    text = _EMOJI.sub("", text)
    text = _LEFTOVER.sub("", text)
    text = _REPEAT_RUN.sub(r"\1", text)
    return _MULTI_SPACE.sub(" ", text).strip()


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
    """Synthesise *text* with gTTS, return an HTML <audio autoplay> tag."""
    if not enabled or not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    snippet = _strip_for_tts(text)[:2000].strip()
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


def tts_text_for_js(text: str) -> str:
    """Return cleaned text ready to pass to the browser Web Speech API."""
    if not isinstance(text, str):
        text = str(text)
    return _strip_for_tts(text)[:2000].strip()


def agent_badges_html(agents_used: list[str]) -> str:
    if not agents_used:
        return ""
    badges = " ".join(
        f'<span class="agent-badge">{_AGENT_LABELS.get(a, a)}</span>'
        for a in agents_used
    )
    return f'<div class="agents-row">Agents: {badges}</div>'
