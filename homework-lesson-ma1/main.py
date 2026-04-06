"""CLI entry point that runs the interactive RAG research agent loop with optional debug output.
Точка входу CLI: інтерактивний цикл RAG-дослідницького агента з опційним виводом debug.

Now supports both single-agent and multi-agent modes.
Тепер підтримує як одноагентний, так і мульти-агентний режими."""

from __future__ import annotations

import argparse
import logging

from agent import agent_multi, AGENT_MULTI_CONFIG    # AGENT_CONFIG, agent


def _extract_last_ai_message(messages: list) -> str:
    """Return the content of the last non-empty AI message in a LangGraph message list.
    Повертає вміст останнього непорожнього повідомлення типу AI зі списку повідомлень LangGraph."""
    for message in reversed(messages):
        message_type = getattr(message, "type", "")
        content = getattr(message, "content", "")
        if message_type == "ai" and content:
            return content
    return "No response generated."


def main() -> None:
    """Run the interactive CLI loop for the RAG agent with optional debug logging.
    Запускає інтерактивний цикл CLI для RAG-агента з опційним журналюванням debug."""
    parser = argparse.ArgumentParser(description="RAG research agent (lesson 5)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log at DEBUG and print full message list (tools + results) each turn",
    )
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Use multi-agent mode with specialized agents (researcher, knowledge expert, report writer)",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    # Select agent mode
    # if args.multi_agent:
    #     selected_agent = agent_multi
    #     selected_config = AGENT_MULTI_CONFIG
    #     mode_name = "Multi-Agent"
    #     print("Multi-Agent RAG Research System (type 'exit' to quit)")
    #     print("Agents: Supervisor, Researcher, Knowledge Expert, Report Writer")
    # else:
    #     selected_agent = agent
    #     selected_config = AGENT_CONFIG
    #     mode_name = "Single-Agent"
    #     print("RAG Research Agent (type 'exit' to quit)")

    selected_agent = agent_multi
    selected_config = AGENT_MULTI_CONFIG
    mode_name = "Multi-Agent"
    print("Multi-Agent RAG Research System (type 'exit' to quit)")
    print("Agents: Supervisor, Researcher, Knowledge Expert, Report Writer")

    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = selected_agent.invoke(
            {"messages": [("user", user_input)]},
            config=selected_config,
        )

        if args.debug:
            print("\n--- debug: messages ---")
            for msg in result.get("messages", []):
                print(msg)
            print("--- end debug ---\n")

        print(f"\n{mode_name} Response: {_extract_last_ai_message(result['messages'])}")


if __name__ == "__main__":
    main()
