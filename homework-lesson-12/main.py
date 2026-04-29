from __future__ import annotations

from config import configure_logging, preview_for_log

configure_logging()

import argparse
import json
import logging
from typing import Any

from langgraph.types import Command

from agent_metrics import get_agent_invoke_counts, reset_agent_invoke_counts
from supervisor import SUPERVISOR_CONFIG, supervisor
from tracing import (
    build_langchain_config,
    default_trace_tags,
    default_user_id,
    flush_langfuse,
    mas_trace,
    observe,
    set_current_trace_io,
)

logger = logging.getLogger(__name__)


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

    logger.info(
        "Supervisor resume: thread_id=%s decision=%s",
        thread_id,
        decision,
    )
    return supervisor.invoke(
        Command(resume={"decisions": [decision]}),
        config=build_langchain_config(
            {"configurable": {"thread_id": thread_id}},
            run_name="supervisor_resume",
        ),
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
        logger.info(
            "HITL interrupt: tool=%s args_preview=%s",
            action.get("tool"),
            preview_for_log(json.dumps(action.get("args", {}), ensure_ascii=False), 600),
        )
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


@observe()
def _run_supervisor_turn(
    *,
    user_input: str,
    thread_id: str,
    session_id: str,
    user_id: str,
    tags: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    with mas_trace(
        trace_name="mas_supervisor_turn",
        session_id=session_id,
        user_id=user_id,
        tags=tags,
    ):
        try:
            set_current_trace_io(input={"user_input": user_input, "thread_id": thread_id})
            result = supervisor.invoke(
                {"messages": [("user", user_input)]},
                config=build_langchain_config(
                    config,
                    run_name="supervisor_agent",
                    extra_metadata={"thread_id": thread_id},
                ),
            )
            final_result = _handle_interrupt(thread_id, result)
            set_current_trace_io(
                output={"final_response": _extract_last_ai_message(final_result.get("messages", []))}
            )
            return final_result
        finally:
            flush_langfuse()


def main() -> None:
    """Runs the interactive REPL for the multi-agent supervisor with HITL on save_report.
    Запускає інтерактивний REPL для мультиагентного супервізора з HITL на save_report."""

    parser = argparse.ArgumentParser(description="Lesson 10 multi-agent supervisor CLI")
    parser.add_argument(
        "--thread-id",
        default="lesson-12-supervisor-cli",
        help="Thread id for checkpointed conversation",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Langfuse session id. Defaults to thread id.",
    )
    parser.add_argument(
        "--user-id",
        default=default_user_id(),
        help="Langfuse user id for trace attribution.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Extra Langfuse trace tag. Repeat to pass multiple tags.",
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

        logger.info(
            "Supervisor invoke: thread_id=%s recursion_limit=%s user=%s",
            args.thread_id,
            config.get("recursion_limit"),
            preview_for_log(user_input),
        )
        reset_agent_invoke_counts()
        try:
            result = _run_supervisor_turn(
                user_input=user_input,
                thread_id=args.thread_id,
                session_id=args.session_id or args.thread_id,
                user_id=args.user_id,
                tags=default_trace_tags() + list(args.tag),
                config=config,
            )
        except Exception:
            logger.exception(
                "Supervisor invoke failed (recursion_limit=%s)",
                config.get("recursion_limit"),
            )
            raise
        logger.debug(
            "Supervisor invoke: raw result keys=%s",
            list(result.keys()) if isinstance(result, dict) else type(result).__name__,
        )

        logger.info(
            "Invoke counts (this user turn): %s",
            get_agent_invoke_counts(),
        )
        print(f"\nAgent: {_extract_last_ai_message(result.get('messages', []))}")


if __name__ == "__main__":
    main()
