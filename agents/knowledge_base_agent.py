"""Knowledge Base Agent — RAG over ChromaDB financial knowledge base."""
import logging
from agents.state import PortfolioAgentState
from services.knowledge_base import query_kb
from services.llm_service import get_llm
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a knowledgeable portfolio management assistant.\n"
     "Answer ONLY using the context below from the financial knowledge base.\n"
     'If the answer is not found in the context, reply exactly: "I don\'t know."\n'
     "Be concise, clear, and use plain language."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def knowledge_base_node(state: PortfolioAgentState) -> dict:
    if "knowledge_base" not in state.get("active_agents", []):
        return {}

    message = state["messages"][-1].content
    docs    = query_kb(message, k=5)

    if not docs:
        return {
            "kb_answer":    "I don't know.",
            "agent_status": state.get("agent_status", []) + ["📚 Knowledge Base: no match"],
        }

    context = "\n\n---\n\n".join(docs)
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
