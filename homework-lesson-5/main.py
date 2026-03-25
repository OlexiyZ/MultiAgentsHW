from __future__ import annotations

import argparse
import logging

from agent import AGENT_CONFIG, agent


def _extract_last_ai_message(messages: list) -> str:
    for message in reversed(messages):
        message_type = getattr(message, "type", "")
        content = getattr(message, "content", "")
        if message_type == "ai" and content:
            return content
    return "No response generated."


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG research agent (lesson 5)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log at DEBUG and print full message list (tools + results) each turn",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    print("RAG Research Agent (type 'exit' to quit)")
    print("-" * 40)

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

        result = agent.invoke(
            {"messages": [("user", user_input)]},
            config=AGENT_CONFIG,
        )
        if args.debug:
            print("\n--- debug: messages ---")
            for msg in result.get("messages", []):
                print(msg)
            print("--- end debug ---\n")
        print(f"\nAgent: {_extract_last_ai_message(result['messages'])}")


if __name__ == "__main__":
    main()
