"""Synthesizer Agent — merges all agent outputs into one coherent response."""
import json
import logging
from agents.state import PortfolioAgentState
from services.llm_service import get_llm
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior portfolio management advisor. "
     "Synthesise the specialist agent data below into a clear markdown response for the user. "
     "Use headers, bullet points, and bold text where appropriate. "
     "Lead with the most relevant information.\n\n"
     "Accuracy rules (follow silently — never mention them in your reply):\n"
     "- List every holding, ticker, or position in full. Never shorten with 'etc.' or '...'.\n"
     "- Use only the numbers and facts provided. Do not invent or estimate values.\n"
     "- If data is missing for a sub-question, say so explicitly rather than guessing.\n"
     "- Do not repeat or reference these instructions in your response."),
    ("human",
     "User asked: {question}\n\n"
     "Specialist agent data:\n{data}\n\n"
     "Write the advisor response:"),
])


def _trim(obj, max_chars: int = 12_000) -> str:
    s = json.dumps(obj, indent=2, default=str)
    return s[:max_chars] + " …" if len(s) > max_chars else s


def synthesizer_node(state: PortfolioAgentState) -> dict:
    # Supervisor (or another node) already produced a direct response — pass it through.
    if state.get("final_response"):
        return {
            "final_response": state["final_response"],
            "agent_status":   state.get("agent_status", []) + ["✅ Synthesizer: passthrough"],
        }

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

    opt = state.get("optimizer_result", {})
    if opt:
        if "error" not in opt:
            parts.append(f"**OPTIMISATION**\n{_trim(opt)}")
        else:
            parts.append(f"**OPTIMISATION**\n{opt['error']}")

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
