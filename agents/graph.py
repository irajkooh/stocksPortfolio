"""
LangGraph multi-agent graph
============================

Flow:
  START
    │
    ▼
  supervisor          ← decides active_agents (incl. optimizer if rebalancing)
    │
    ▼
  market_intel        ← self-skips if not in active_agents
    │
    ▼
  portfolio_analyst   ← self-skips if not in active_agents
    │
    ▼
  risk_manager        ← self-skips if not in active_agents
    │
    ▼
  optimizer           ← self-skips if not in active_agents
    │                    (auto-triggered for rebalancing intent)
    ▼
  knowledge_base      ← self-skips if not in active_agents
    │
    ▼
  synthesizer         ← always runs, merges all outputs
    │
    ▼
  END
"""
import logging
from functools import lru_cache

from langgraph.graph import StateGraph, END

from agents.state import PortfolioAgentState
from agents.supervisor         import supervisor_node
from agents.market_intel       import market_intel_node
from agents.portfolio_analyst  import portfolio_analyst_node
from agents.risk_manager       import risk_manager_node
from agents.optimizer_agent import optimizer_node
from agents.knowledge_base_agent import knowledge_base_node
from agents.synthesizer        import synthesizer_node

logger = logging.getLogger(__name__)


def _build() -> "CompiledGraph":
    g = StateGraph(PortfolioAgentState)

    g.add_node("supervisor",        supervisor_node)
    g.add_node("market_intel",      market_intel_node)
    g.add_node("portfolio_analyst", portfolio_analyst_node)
    g.add_node("risk_manager",      risk_manager_node)
    g.add_node("optimizer",          optimizer_node)
    g.add_node("knowledge_base",    knowledge_base_node)
    g.add_node("synthesizer",       synthesizer_node)

    g.set_entry_point("supervisor")
    g.add_edge("supervisor",        "market_intel")
    g.add_edge("market_intel",      "portfolio_analyst")
    g.add_edge("portfolio_analyst", "risk_manager")
    g.add_edge("risk_manager",      "optimizer")
    g.add_edge("optimizer",         "knowledge_base")
    g.add_edge("knowledge_base",    "synthesizer")
    g.add_edge("synthesizer",       END)

    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    logger.info("Compiling LangGraph agent graph …")
    return _build()
