"""
Supervisor Agent
----------------
Reads the user message and decides which specialist agents to invoke.
Optimizer triggers automatically when rebalancing / optimisation intent
is detected — no explicit button click required.
"""
import json
import logging
import re

from agents.state import PortfolioAgentState
from services.llm_service import get_llm
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# ── Greetings and off-topic short-circuit ─────────────────────────────────────
_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings", "sup", "yo",
    "good morning", "good afternoon", "good evening", "good day",
    "how are you", "how are you doing", "how are you today",
    "how's it going", "hows it going", "how is it going",
    "what's up", "whats up", "what is up",
    "how do you do", "nice to meet you", "pleased to meet you",
    "how are things", "how have you been", "how you doing",
}
_GREETING_REPLY = (
    "Hi! I'm your AI portfolio management assistant. "
    "Ask me about your holdings, stock prices, risk metrics, or investment strategies — "
    "I'm happy to help."
)

# ── Keyword fallback (used if LLM JSON parse fails) ──────────────────────────
_KEYWORD_MAP = {
    "market_intel": [
        "price", "stock", "quote", "market", "ticker", "today",
        "trading", "52 week", "pe ratio", "earnings",
    ],
    "portfolio_analyst": [
        "portfolio", "holding", "p&l", "profit", "loss", "position",
        "allocation", "performance", "my stocks", "how am i doing",
        "what do i own", "what do i have", "my positions", "my holdings",
        "what are my", "show me my", "list my",
    ],
    "risk_manager": [
        "risk", "sharpe", "var", "drawdown", "volatility", "beta",
        "safe", "exposure", "sortino", "downside",
    ],
    "optimizer": [
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
- market_intel       : live stock prices, company info, market data for specific tickers
- portfolio_analyst  : current portfolio holdings, P&L, allocation, positions, what the user owns
- risk_manager       : Sharpe ratio, VaR, drawdown, volatility, beta, risk analysis of the user's portfolio
- optimizer          : Markowitz mean-variance optimisation, rebalancing, budget allocation
- knowledge_base     : conceptual finance / investing questions (definitions, theory, formulas, how metrics work)

Return ONLY valid JSON, no other text:
{{"agents": ["agent1", "agent2"], "intent": "one-line summary"}}

Routing rules (follow exactly):
• "what are my holdings / positions / stocks" → portfolio_analyst ONLY
• "what do I own / have in my portfolio" → portfolio_analyst ONLY
• "how am I doing / portfolio performance / P&L / profit / loss" → portfolio_analyst + risk_manager
• Specific ticker symbols mentioned → always include market_intel
• Rebalance / optimise / allocate budget / suggest best weights → always include optimizer
• "rebalance for maximum Sharpe" / "optimise for Sharpe ratio X" → optimizer + knowledge_base
• "what is X / explain X / define X / how does X work" with no portfolio context → knowledge_base ONLY
• Examples of knowledge_base ONLY questions: "What is VaR?", "Explain Sharpe ratio", "What is Sortino ratio?", "How does Markowitz optimisation work?"
• Portfolio health check (no specific question) → portfolio_analyst + risk_manager
• Always include at least one agent.
• Do NOT include knowledge_base for questions about the user's own portfolio or holdings.
• Do NOT include market_intel unless a specific ticker or company name is mentioned.
• Greetings (hi, hello, hey, etc.) or completely off-topic questions (weather, science, cooking) → knowledge_base ONLY.
• Do NOT route greetings or off-topic messages to portfolio_analyst or risk_manager.
"""),
    ("human", "{message}"),
])


def _keyword_fallback(message: str) -> list[str]:
    msg = message.lower()
    # Conceptual questions ("what is X", "explain X") → knowledge_base only,
    # even if the subject word (sharpe, var, etc.) also hits risk_manager.
    kb_triggers = _KEYWORD_MAP["knowledge_base"]
    if any(k in msg for k in kb_triggers):
        agents = ["knowledge_base"]
        # If there's also an optimisation intent, add optimizer
        if any(k in msg for k in _KEYWORD_MAP["optimizer"]):
            agents.append("optimizer")
        return agents
    agents = []
    for agent, keywords in _KEYWORD_MAP.items():
        if agent == "knowledge_base":
            continue
        if any(k in msg for k in keywords):
            agents.append(agent)
    return agents or ["knowledge_base"]


def supervisor_node(state: PortfolioAgentState) -> dict:
    message = state["messages"][-1].content
    stripped = message.strip().lower().rstrip("!.?,")
    if stripped in _GREETINGS:
        return {
            "active_agents": [],
            "user_intent":   "greeting",
            "agent_status":  ["💬 Greeting — responding directly"],
            "final_response": _GREETING_REPLY,
        }

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
