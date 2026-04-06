from __future__ import annotations

import argparse
import json
from typing import Any

from langgraph.types import Command

from supervisor import SUPERVISOR_CONFIG, supervisor


def _extract_last_ai_message(messages: list[Any]) -> str:
    """Returns the content of the last non-empty AI message or a default placeholder.
    Повертає вміст останнього непорожнього повідомлення AI або текст за замовчуванням."""

    for message in reversed(messages):
        if getattr(message, "type", "") == "ai" and getattr(message, "content", ""):
            return message.content
    return "No response generated."


def _interrupt_payload(result: dict[str, Any]) -> Any | None:
    """Extracts the first HITL interrupt object from a supervisor invoke result, if any.
    Дістає перший об'єкт переривання HITL з результату виклику супервізора, якщо він є."""

    interrupts = result.get("__interrupt__")
    if interrupts:
        return interrupts[0]
    return None


def _extract_action_data(interrupt: Any) -> dict[str, Any]:
    """Normalizes an interrupt payload into a dict with tool name and arguments for display.
    Нормалізує вміст переривання у словник із назвою інструмента та аргументами для показу."""

    payload = getattr(interrupt, "value", interrupt)
    if isinstance(payload, dict):
        action_requests = payload.get("action_requests") or []
        if action_requests:
            req = action_requests[0]
            return {
                "tool": req.get("name", "save_report"),
                "args": req.get("args", {}),
            }
        return {
            "tool": payload.get("tool", "save_report"),
            "args": payload.get("args", {}),
        }
    return {"tool": "save_report", "args": {}}


def _print_pending_action(action_data: dict[str, Any]) -> None:
    """Prints a human-readable approval banner with tool name and JSON args.
    Виводить зрозумілий банер затвердження з назвою інструмента та JSON-аргументами."""

    print("\n" + "=" * 60)
    print("ACTION REQUIRES APPROVAL")
    print("=" * 60)
    print(f"Tool: {action_data.get('tool', 'save_report')}")
    print("Args:")
    print(json.dumps(action_data.get("args", {}), ensure_ascii=False, indent=2))


def _resume_with_decision(
    thread_id: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Resumes the checkpointed supervisor graph with a single HITL decision payload.
    Відновлює граф супервізора з checkpointer за одним рішенням HITL."""

    return supervisor.invoke(
        Command(resume={"decisions": [decision]}),
        config={"configurable": {"thread_id": thread_id}},
    )


def _handle_interrupt(thread_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """Loops on HITL interrupts, prompting approve/edit/reject until the run completes.
    У циклі обробляє переривання HITL із запитом approve/edit/reject, доки виконання не завершиться."""

    current = result
    while True:
        interrupt = _interrupt_payload(current)
        if not interrupt:
            return current

        action = _extract_action_data(interrupt)
        _print_pending_action(action)

        while True:
            user_action = input("\napprove / edit / reject: ").strip().lower()
            if user_action in {"approve", "edit", "reject"}:
                break
            print("Please type: approve, edit, or reject")

        if user_action == "approve":
            current = _resume_with_decision(thread_id, {"type": "approve"})
            continue

        if user_action == "edit":
            feedback = input("Your feedback: ").strip()
            current = _resume_with_decision(
                thread_id,
                {
                    "type": "edit",
                    "edited_action": {"feedback": feedback},
                },
            )
            continue

        current = _resume_with_decision(
            thread_id,
            {
                "type": "reject",
                "message": "User rejected save_report action.",
            },
        )


def main() -> None:
    """Runs the interactive REPL for the multi-agent supervisor with HITL on save_report.
    Запускає інтерактивний REPL для мультиагентного супервізора з HITL на save_report."""

    parser = argparse.ArgumentParser(description="Lesson 8 multi-agent supervisor CLI")
    parser.add_argument(
        "--thread-id",
        default="lesson-8-supervisor-cli",
        help="Thread id for checkpointed conversation",
    )
    args = parser.parse_args()

    print("Multi-agent Research Supervisor (type 'exit' to quit)")
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

        config = dict(SUPERVISOR_CONFIG)
        config["configurable"] = {"thread_id": args.thread_id}

        result = supervisor.invoke(
            {"messages": [("user", user_input)]},
            config=config,
        )

        result = _handle_interrupt(args.thread_id, result)
        print(f"\nAgent: {_extract_last_ai_message(result.get('messages', []))}")


if __name__ == "__main__":
    main()
