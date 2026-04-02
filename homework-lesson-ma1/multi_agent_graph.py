"""Multi-agent LangGraph workflow with supervisor pattern.
Мульти-агентний LangGraph workflow з патерном супервайзера."""

from __future__ import annotations

import json
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from config import Settings
from graph_agents import (
    SUPERVISOR_PROMPT,
    researcher_node,
    knowledge_expert_node,
    report_writer_node,
)
from tools import web_search, read_url, knowledge_search, write_report


settings = Settings()


class AgentState(TypedDict):
    """State schema for the multi-agent graph.
    Схема стану для мульти-агентного графа."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    sender: str


def create_supervisor_chain():
    """Create the supervisor LLM chain that routes to agents.
    Створює ланцюжок LLM супервайзера, який направляє до агентів."""
    llm = ChatOpenAI(
        api_key=settings.api_key.get_secret_value(),
        model=settings.model_name,
        timeout=settings.request_timeout,
        temperature=0,
    )
    return llm


def supervisor_node(state: AgentState) -> dict:
    """Supervisor node that decides which agent should act next.
    Вузол супервайзера, який вирішує, який агент має діяти наступним."""
    messages = state.get("messages", [])

    supervisor_llm = create_supervisor_chain()

    # Prepare context for supervisor
    context = f"{SUPERVISOR_PROMPT}\n\nПоточна історія діалогу та результати роботи агентів:\n"
    for msg in messages[-5:]:  # Last 5 messages for context
        if isinstance(msg, HumanMessage):
            context += f"Користувач: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            context += f"Агент: {msg.content}\n"

    context += "\n\nВирішуй, який агент має діяти далі або чи завершено роботу."

    response = supervisor_llm.invoke([HumanMessage(content=context)])

    # Parse supervisor decision
    try:
        # Try to extract JSON from response
        content = response.content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content

        decision = json.loads(json_str)
        next_agent = decision.get("next_agent", "FINISH")
        task = decision.get("task", "")

    except (json.JSONDecodeError, IndexError):
        # If parsing fails, try to infer from content
        content_lower = response.content.lower()
        if "researcher" in content_lower:
            next_agent = "researcher"
            task = "Знайти інформацію в Інтернеті"
        elif "knowledge" in content_lower:
            next_agent = "knowledge_expert"
            task = "Знайти інформацію в базі знань"
        elif "report" in content_lower or "writer" in content_lower:
            next_agent = "report_writer"
            task = "Створити звіт"
        else:
            next_agent = "FINISH"
            task = ""

    # Add task as a message if next agent is not FINISH
    if next_agent != "FINISH" and task:
        return {
            "messages": [AIMessage(content=f"[Супервайзер → {next_agent}]: {task}")],
            "next_agent": next_agent,
            "sender": "supervisor"
        }

    return {
        "messages": [],
        "next_agent": next_agent,
        "sender": "supervisor"
    }


def researcher_agent_node(state: AgentState) -> dict:
    """Researcher agent node with tool execution.
    Вузол агента-дослідника з виконанням інструментів."""
    result = researcher_node(state)

    # Check if tools were called
    messages = result.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        # Tools will be executed in the tool node
        result["next_agent"] = "tools"
    else:
        # No tools called, go back to supervisor
        result["next_agent"] = "supervisor"

    return result


def knowledge_expert_agent_node(state: AgentState) -> dict:
    """Knowledge expert agent node with tool execution.
    Вузол агента-експерта з бази знань з виконанням інструментів."""
    result = knowledge_expert_node(state)

    # Check if tools were called
    messages = result.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        result["next_agent"] = "tools"
    else:
        result["next_agent"] = "supervisor"

    return result


def report_writer_agent_node(state: AgentState) -> dict:
    """Report writer agent node with tool execution.
    Вузол агента-письменника звітів з виконанням інструментів."""
    result = report_writer_node(state)

    # Check if tools were called
    messages = result.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        result["next_agent"] = "tools"
    else:
        result["next_agent"] = "supervisor"

    return result


def router(state: AgentState) -> str:
    """Route to the next node based on state.
    Маршрутизація до наступного вузла на основі стану."""
    next_agent = state.get("next_agent", "supervisor")

    if next_agent == "FINISH":
        return END

    return next_agent


def create_multi_agent_graph():
    """Create and compile the multi-agent LangGraph workflow.
    Створює та компілює мульти-агентний LangGraph workflow."""

    # Create tool node with all tools
    all_tools = [web_search, read_url, knowledge_search, write_report]
    tool_node = ToolNode(all_tools)

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_agent_node)
    workflow.add_node("knowledge_expert", knowledge_expert_agent_node)
    workflow.add_node("report_writer", report_writer_agent_node)
    workflow.add_node("tools", tool_node)

    # Add edges
    workflow.set_entry_point("supervisor")

    # Supervisor can route to any agent
    workflow.add_conditional_edges(
        "supervisor",
        router,
        {
            "researcher": "researcher",
            "knowledge_expert": "knowledge_expert",
            "report_writer": "report_writer",
            END: END,
        }
    )

    # Each agent can go to tools or back to supervisor
    for agent in ["researcher", "knowledge_expert", "report_writer"]:
        workflow.add_conditional_edges(
            agent,
            router,
            {
                "tools": "tools",
                "supervisor": "supervisor",
            }
        )

    # Tools always go back to supervisor
    workflow.add_edge("tools", "supervisor")

    # Compile with checkpointer for memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Create the graph instance
multi_agent_graph = create_multi_agent_graph()


# Configuration for the graph
GRAPH_CONFIG = {
    "configurable": {"thread_id": "multi-agent-research-cli"},
    "recursion_limit": settings.max_iterations,
}
