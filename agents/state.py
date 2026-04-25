import operator
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage


class PortfolioAgentState(TypedDict):
    # conversation
    messages:            Annotated[list[BaseMessage], operator.add]
    # supervisor decisions
    user_intent:         str
    active_agents:       list[str]
    agent_status:        list[str]
    # portfolio context
    active_portfolio_id: int        # which named portfolio to query
    # agent outputs
    market_data:         dict
    portfolio_data:      dict
    risk_metrics:        dict
    optimizer_result:    dict
    kb_answer:           str
    # final
    final_response:      str
    charts:              list       # plotly Figure objects


def empty_state(message: str, portfolio_id: int = 1) -> PortfolioAgentState:
    from langchain_core.messages import HumanMessage
    return PortfolioAgentState(
        messages             = [HumanMessage(content=message)],
        user_intent          = "",
        active_agents        = [],
        agent_status         = [],
        active_portfolio_id  = portfolio_id,
        market_data          = {},
        portfolio_data       = {},
        risk_metrics         = {},
        optimizer_result     = {},
        kb_answer            = "",
        final_response       = "",
        charts               = [],
    )
