"""Knowledge Base Agent — RAG over ChromaDB financial knowledge base."""
import logging
from agents.state import PortfolioAgentState
from services.knowledge_base import query_kb
from services.llm_service import get_llm
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial knowledge assistant. Answer ONLY finance and investment questions.\n"
     "Use the knowledge base context below when it is relevant.\n"
     "If the context is empty or unhelpful, you may still answer standard finance concepts "
     "(e.g. VaR, Sharpe ratio, Sortino ratio, Markowitz optimisation, beta, drawdown) "
     "from general knowledge — but only if the question is genuinely about finance or investing.\n\n"
     "STRICT RULES (never break these):\n"
     "- NEVER invent portfolio holdings, ticker symbols, allocations, fees, or any specific financial data.\n"
     "- If the question is not about finance or investing, reply with exactly: "
     "'I can only help with finance and investment questions. Try asking about your portfolio, "
     "stock prices, risk metrics, or investment concepts.'\n"
     "- Do not attempt to answer general-knowledge, science, or personal questions."),
    ("human", "Knowledge base context (may be empty or partial):\n{context}\n\nQuestion: {question}"),
])


def knowledge_base_node(state: PortfolioAgentState) -> dict:
    if "knowledge_base" not in state.get("active_agents", []):
        return {}

    message = state["messages"][-1].content
    docs    = query_kb(message, k=5)
    context = "\n\n---\n\n".join(docs) if docs else "(no matching entries in knowledge base)"

    try:
        llm    = get_llm()
        result = (_PROMPT | llm).invoke({"context": context, "question": message})
        answer = result.content if hasattr(result, "content") else str(result)
        answer = answer.strip() or "I don't know."
    except Exception as exc:
        logger.error("KB LLM error: %s", exc)
        answer = "I don't know."

    status = "📚 Knowledge Base: answer found" if answer != "I don't know." else "📚 Knowledge Base: no answer"
    logger.info(status)
    return {
        "kb_answer":    answer,
        "agent_status": state.get("agent_status", []) + [status],
    }
