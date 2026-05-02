from __future__ import annotations

from tracing import build_langchain_config


def test_build_langchain_config_propagates_langfuse_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        "tracing.current_trace_attributes",
        lambda: {
            "trace_name": "mas_supervisor_turn",
            "session_id": "session-123",
            "user_id": "user-456",
            "tags": ["mas", "lesson-12"],
            "enabled": False,
        },
    )

    config = build_langchain_config(
        {"configurable": {"thread_id": "thread-1"}},
        run_name="planner_agent",
        extra_metadata={"subagent": "planner"},
    )

    assert config["run_name"] == "planner_agent"
    assert config["configurable"] == {"thread_id": "thread-1"}
    assert config["metadata"]["langfuse_session_id"] == "session-123"
    assert config["metadata"]["langfuse_user_id"] == "user-456"
    assert config["metadata"]["langfuse_tags"] == ["mas", "lesson-12"]
    assert config["metadata"]["subagent"] == "planner"
    assert "callbacks" not in config


def test_build_langchain_config_preserves_existing_metadata(monkeypatch) -> None:
    monkeypatch.setattr("tracing.current_trace_attributes", lambda: None)

    config = build_langchain_config(
        {"metadata": {"already": "there"}},
        run_name="research_agent",
        extra_metadata={"new": "value"},
    )

    assert config["run_name"] == "research_agent"
    assert config["metadata"] == {"already": "there", "new": "value"}
