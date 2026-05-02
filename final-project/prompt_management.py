from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

try:
    from langfuse import Langfuse
except ImportError:  # pragma: no cover
    Langfuse = None  # type: ignore[assignment]

    def _build_client() -> Any:
        raise RuntimeError(
            "Langfuse SDK is required for Prompt Management. Install the 'langfuse' package."
        )

from config import Settings


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _build_client() -> Any:
    """Builds a Langfuse client from project settings and caches it for prompt fetches.
    Створює клієнт Langfuse з налаштувань проєкту та кешує його для завантаження промптів."""

    settings = Settings()
    return Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value()
        if settings.langfuse_public_key
        else None,
        secret_key=settings.langfuse_secret_key.get_secret_value()
        if settings.langfuse_secret_key
        else None,
        base_url=settings.langfuse_base_url,
    )


@lru_cache(maxsize=None)
def _load_prompt(name: str, label: str) -> Any:
    """Loads a prompt object from Langfuse by name and label and caches it.
    Завантажує об'єкт промпту з Langfuse за іменем і label та кешує його."""

    logger.info("Loading Langfuse prompt name=%s label=%s", name, label)
    return _build_client().get_prompt(name, label=label)


def load_system_prompt(name: str, label: str, **variables: Any) -> str:
    """Fetches a text prompt from Langfuse and compiles it into a system prompt string.
    Отримує текстовий промпт із Langfuse та компілює його в рядок system prompt."""

    prompt = _load_prompt(name, label)
    compiled = prompt.compile(**variables)
    if not isinstance(compiled, str):
        raise TypeError(
            f"Langfuse prompt {name!r} with label {label!r} must compile to text, "
            f"got {type(compiled).__name__}"
        )
    return compiled


def get_planner_system_prompt(settings: Settings | None = None) -> str:
    """Resolves the Planner system prompt from Langfuse using configured prompt metadata.
    Завантажує system prompt планувальника з Langfuse за конфігурацією проєкту."""

    settings = settings or Settings()
    return load_system_prompt(
        settings.planner_prompt_name,
        settings.langfuse_prompt_label,
    )


def get_research_system_prompt(settings: Settings | None = None) -> str:
    """Resolves the Research system prompt from Langfuse using configured prompt metadata.
    Завантажує system prompt дослідника з Langfuse за конфігурацією проєкту."""

    settings = settings or Settings()
    return load_system_prompt(
        settings.research_prompt_name,
        settings.langfuse_prompt_label,
    )


def get_critic_system_prompt(settings: Settings | None = None) -> str:
    """Resolves the Critic system prompt from Langfuse using configured prompt metadata.
    Завантажує system prompt критика з Langfuse за конфігурацією проєкту."""

    settings = settings or Settings()
    return load_system_prompt(
        settings.critic_prompt_name,
        settings.langfuse_prompt_label,
    )


def get_supervisor_system_prompt(settings: Settings | None = None) -> str:
    """Resolves the Supervisor system prompt from Langfuse using configured prompt metadata.
    Завантажує system prompt супервізора з Langfuse за конфігурацією проєкту."""

    settings = settings or Settings()
    return load_system_prompt(
        settings.supervisor_prompt_name,
        settings.langfuse_prompt_label,
    )
