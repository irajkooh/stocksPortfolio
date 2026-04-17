"""
Supervisor Agent
----------------
Reads the user message and decides which specialist agents to invoke.
RL Optimizer triggers automatically when rebalancing / optimisation intent
is detected — no explicit button click required.
"""
import json
import logging
import re

from agents.state import PortfolioAgentState
from services.llm_service import get_llm
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# ── Keyword fallback (used if LLM JSON parse fails) ──────────────────────────
_KEYWORD_MAP = {
    "market_intel": [
        "price", "stock", "quote", "market", "ticker", "today",
        "trading", "52 week", "pe ratio", "earnings",
    ],
    "portfolio_analyst": [
        "portfolio", "holding", "p&l", "profit", "loss", "position",
        "allocation", "performance", "my stocks", "how am i doing",
    ],
    "risk_manager": [
        "risk", "sharpe", "var", "drawdown", "volatility", "beta",
        "safe", "exposure", "sortino", "downside",
    ],
    "rl_optimizer": [
        "optimis", "optimiz", "rebalanc", "weight", "allocat",
        "budget", "invest", "suggest", "recommend", "best allocation",
        "how should i invest", "what should i buy",
    ],
    "knowledge_base": [
        "what is", "explain", "how does", "define", "tell me about",
        "meaning of", "concept", "theory", "formula",
    ],
}

_SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a portfolio management supervisor.
Analyse the user request and decide which specialist agents to invoke.

AGENTS:
- market_intel       : live stock prices, company info, market data
- portfolio_analyst  : current portfolio P&L, allocation, positions
- risk_manager       : Sharpe ratio, VaR, drawdown, volatility, beta
- rl_optimizer       : RL-based weight optimisation, rebalancing, budget allocation
                       → ALWAYS include when user mentions: rebalance, optimise,
                         allocate budget, best weights, what should I buy/invest
- knowledge_base     : conceptual finance / investing questions (RAG)

Return ONLY valid JSON, no other text:
{{"agents": ["agent1", "agent2"], "intent": "one-line summary"}}

Rules:
• Include rl_optimizer whenever the request is about rebalancing or optimisation.
• Include portfolio_analyst + risk_manager together when user asks about portfolio health.
• Include market_intel whenever specific tickers are mentioned.
• Always include at least one agent.
"""),
    ("human", "{message}"),
])


def _keyword_fallback(message: str) -> list[str]:
    msg = message.lower()
    agents = []
    for agent, keywords in _KEYWORD_MAP.items():
        if any(k in msg for k in keywords):
            agents.append(agent)
    return agents or ["knowledge_base"]


def supervisor_node(state: PortfolioAgentState) -> dict:
    message = state["messages"][-1].content
    llm = get_llm()

    try:
        chain  = _SUPERVISOR_PROMPT | llm
        result = chain.invoke({"message": message})
        text   = result.content if hasattr(result, "content") else str(result)

        # extract first JSON object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("No JSON in response")
        parsed = json.loads(m.group())
        agents = parsed.get("agents", [])
        intent = parsed.get("intent", "")

        # validate agent names
        valid  = set(_KEYWORD_MAP.keys())
        agents = [a for a in agents if a in valid]
        if not agents:
            agents = _keyword_fallback(message)

    except Exception as exc:
        logger.warning("Supervisor LLM parse failed (%s) — using keyword fallback", exc)
        agents = _keyword_fallback(message)
        intent = "keyword-routed request"

    logger.info("Supervisor → agents: %s", agents)
    return {
        "active_agents": agents,
        "user_intent":   intent,
        "agent_status":  [f"🎯 Routing to: {', '.join(agents)}"],
    }
