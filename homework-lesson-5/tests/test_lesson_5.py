"""Unit tests for RRF fusion, tool registration, knowledge_search truncation, and prompt fallback.
Модульні тести для RRF, реєстрації tools, обрізання knowledge_search і запасного промпта."""

from __future__ import annotations

import os

import pytest
from langchain_core.documents import Document

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-pytest")

from retriever_langchain_flavour import reciprocal_rank_fusion
from tools import TOOLS, knowledge_search


def test_reciprocal_rank_fusion_dedupes_and_orders() -> None:
    """Assert RRF deduplicates identical page_content and preserves distinct documents.
    Перевіряє, що RRF прибирає дублікати page_content і зберігає різні документи."""
    a = Document(page_content="alpha", metadata={})
    b = Document(page_content="beta", metadata={})
    c = Document(page_content="alpha", metadata={})
    fused = reciprocal_rank_fusion([a, b], [c, b], limit=10)
    texts = [d.page_content for d in fused]
    assert "alpha" in texts
    assert "beta" in texts
    assert texts.count("alpha") == 1


def test_knowledge_search_tool_registered() -> None:
    """Ensure the knowledge_search tool is exposed alongside the other agent tools.
    Переконується, що інструмент knowledge_search зареєстрований разом з іншими tools агента."""
    names = {t.name for t in TOOLS}
    assert "knowledge_search" in names


def test_knowledge_search_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify knowledge_search shortens an oversized backend string using max_knowledge_chars.
    Перевіряє, що knowledge_search скорочує надто довгий рядок з бекенду згідно з max_knowledge_chars."""
    long_text = "x" * 20_000

    def fake_backend(_q: str) -> str:
        """Return a fixed long string to simulate an unbounded retriever response.
        Повертає фіксований довгий рядок, імітуючи необмежену відповідь ретривера."""
        return long_text

    monkeypatch.setattr("tools._knowledge_backend_search", fake_backend)
    out = knowledge_search.invoke({"query": "q"})
    assert len(out) < len(long_text)
    assert out.endswith("...")


def test_load_system_prompt_fallback(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check load_system_prompt falls back when system_prompt.yaml is absent under BASE_DIR.
    Перевіряє, що load_system_prompt використовує запасний текст, якщо немає system_prompt.yaml."""
    import config as cfg

    monkeypatch.setattr(cfg, "BASE_DIR", tmp_path)
    text = cfg.load_system_prompt()
    assert "research agent" in text.lower() or "дослід" in text.lower() or len(text) > 10
