from __future__ import annotations

from prompt_management import _build_client, _load_prompt, load_system_prompt


class _FakePrompt:
    def __init__(self, compiled: object) -> None:
        self._compiled = compiled
        self.variables = None

    def compile(self, **variables: object) -> object:
        self.variables = variables
        return self._compiled


class _FakeClient:
    def __init__(self, prompt: _FakePrompt) -> None:
        self.prompt = prompt
        self.calls: list[tuple[str, str]] = []

    def get_prompt(self, name: str, label: str):
        self.calls.append((name, label))
        return self.prompt


def test_load_system_prompt_compiles_langfuse_prompt(monkeypatch) -> None:
    fake_prompt = _FakePrompt("compiled prompt")
    fake_client = _FakeClient(fake_prompt)
    monkeypatch.setattr("prompt_management._build_client", lambda: fake_client)
    _build_client.cache_clear()
    _load_prompt.cache_clear()

    result = load_system_prompt(
        "final-project/planner-system",
        "production",
        user_role="planner",
    )

    assert result == "compiled prompt"
    assert fake_client.calls == [("final-project/planner-system", "production")]
    assert fake_prompt.variables == {"user_role": "planner"}


def test_load_system_prompt_rejects_non_text_prompt(monkeypatch) -> None:
    fake_client = _FakeClient(_FakePrompt(["not", "text"]))
    monkeypatch.setattr("prompt_management._build_client", lambda: fake_client)
    _build_client.cache_clear()
    _load_prompt.cache_clear()

    try:
        load_system_prompt("final-project/planner-system", "production")
    except TypeError as exc:
        assert "must compile to text" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected TypeError for non-text compiled prompt")
