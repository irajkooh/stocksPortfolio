"""Synthesizer Agent — merges all agent outputs into one coherent response."""
import json
import logging
from agents.state import PortfolioAgentState
from services.llm_service import get_llm
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior portfolio management advisor synthesising reports from specialist agents.\n"
     "Write a clear, well-structured markdown response for the user.\n"
     "Use headers, bullet points and bold text where appropriate.\n"
     "Lead with the most relevant information. Be concise but complete."),
    ("human",
     "User asked: {question}\n\n"
     "Specialist agent data:\n{data}\n\n"
     "Write the synthesised advisor response:"),
])


def _trim(obj, max_chars: int = 1_200) -> str:
    s = json.dumps(obj, indent=2, default=str)
    return s[:max_chars] + " …" if len(s) > max_chars else s


def synthesizer_node(state: PortfolioAgentState) -> dict:
    message = state["messages"][-1].content
    parts: list[str] = []

    if state.get("market_data"):
        parts.append(f"**MARKET DATA**\n{_trim(state['market_data'])}")

    pd = state.get("portfolio_data", {})
    if pd and not pd.get("empty"):
        parts.append(f"**PORTFOLIO**\n{_trim(pd)}")

    rm = state.get("risk_metrics", {})
    if rm and "error" not in rm:
        parts.append(f"**RISK METRICS**\n{_trim(rm)}")

    rl = state.get("rl_result", {})
    if rl and "error" not in rl:
        parts.append(f"**RL OPTIMISATION**\n{_trim(rl)}")

    kb = state.get("kb_answer", "")
    if kb and kb != "I don't know.":
        parts.append(f"**KNOWLEDGE BASE**\n{kb}")

    if not parts:
        # Nothing useful found — propagate KB "I don't know"
        return {
            "final_response": (
                kb if kb else
                "I couldn't find relevant information to answer your question."
            ),
            "agent_status": state.get("agent_status", []) + ["✅ Synthesizer: done"],
        }

    combined = "\n\n".join(parts)
    try:
        llm      = get_llm()
        result   = (_PROMPT | llm).invoke({"question": message, "data": combined})
        response = result.content if hasattr(result, "content") else str(result)
    except Exception as exc:
        logger.error("Synthesizer LLM error: %s", exc)
        response = combined        # graceful fallback: return raw data

    return {
        "final_response": response.strip(),
        "agent_status":   state.get("agent_status", []) + ["✅ Synthesizer: done"],
    }
