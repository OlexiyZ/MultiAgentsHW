"""Multi-agent LangGraph graph for support workflow.
Мульти-агентний LangGraph граф для support workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from config import Settings
from graph_agents import (
    request_analyzer_agent,
    goods_finder_agent,
    offer_generator_agent,
)


settings = Settings()


class SupportState(TypedDict):
    """State schema for the support multi-agent graph.
    Схема стану для підтримки мульти-агентного графа."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    sender: str
    request_id: str
    sender_email: str
    sender_contact_id: str
    sender_company_id: str
    responsible_user_id: str
    requested_goods: list[dict]
    found_goods: list[dict]
    offer_text: str


def request_analyzer_node(state: SupportState) -> dict:
    """Request analyzer agent node.
    Вузол агента-аналізатора запиту."""
    return request_analyzer_agent(state)


def goods_finder_node(state: SupportState) -> dict:
    """Goods finder agent node.
    Вузол агента-пошуку товарів."""
    return goods_finder_agent(state)


def offer_generator_node(state: SupportState) -> dict:
    """Offer generator agent node.
    Вузол агента-генератора пропозицій."""
    return offer_generator_agent(state)




def create_multi_agent_graph():
    """Create and compile the multi-agent LangGraph graph.
    Створює та компілює мульти-агентний LangGraph graph."""

    graph = StateGraph(SupportState)

    # Add nodes
    graph.add_node("request_analyzer", request_analyzer_node)
    graph.add_node("goods_finder", goods_finder_node)
    graph.add_node("offer_generator", offer_generator_node)

    # Add edges
    graph.add_edge(START, "request_analyzer")
    graph.add_edge("request_analyzer", "goods_finder")
    graph.add_edge("goods_finder", "offer_generator")
    graph.add_edge("offer_generator", END)


    # Compile with checkpointer for memory
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Create the graph instance
multi_agent_graph = create_multi_agent_graph()


# Configuration for the graph
GRAPH_CONFIG = {
    "configurable": {"thread_id": "multi-agent-research-cli"},
    "recursion_limit": settings.max_iterations,
}
