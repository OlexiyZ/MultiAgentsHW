"""Per–user-turn counters: supervisor tools (plan/research/critique/save_report) and sub-agents."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

_counts: defaultdict[str, int] = defaultdict(int)


def reset_agent_invoke_counts() -> None:
    _counts.clear()


def record_agent_invoke(agent_name: str) -> None:
    _counts[agent_name] += 1


def record_supervisor_tool(tool_name: str) -> None:
    """Count a supervisor-level tool run (plan / research / critique / save_report)."""

    _counts[f"supervisor.{tool_name}"] += 1


def get_agent_invoke_counts() -> Mapping[str, int]:
    return dict(_counts)
