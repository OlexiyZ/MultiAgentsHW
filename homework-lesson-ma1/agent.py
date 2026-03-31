"""Wires the chat model, tools, memory checkpointer, and LangGraph agent for the homework app.
Зв'язує чат-модель, інструменти, checkpointer пам'яті та LangGraph-агента для домашнього застосунку.

This module now supports both single-agent and multi-agent modes.
Цей модуль тепер підтримує як одноагентний, так і мульти-агентний режими."""

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
# from langgraph.prebuilt import create_react_agent

from config import SYSTEM_PROMPT, Settings
from tools import TOOLS


settings = Settings()

llm = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.model_name,
    timeout=settings.request_timeout,
)

memory = MemorySaver()

# Single agent (legacy mode) - для сумісності з існуючим кодом
agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)


AGENT_CONFIG = {
    "configurable": {"thread_id": "research-agent-cli"},
    "recursion_limit": settings.max_iterations,
}


# Multi-agent mode - import the graph
try:
    from multi_agent_graph import multi_agent_graph, GRAPH_CONFIG

    # Use multi-agent graph as the default agent
    agent_multi = multi_agent_graph
    AGENT_MULTI_CONFIG = GRAPH_CONFIG
except ImportError:
    # Fallback to single agent if multi-agent is not available
    agent_multi = agent
    AGENT_MULTI_CONFIG = AGENT_CONFIG
