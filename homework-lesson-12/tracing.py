from __future__ import annotations

import getpass
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any, Callable, Iterator

from config import Settings

try:
    from langfuse import Langfuse, get_client, observe, propagate_attributes
    from langfuse.langchain import CallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    Langfuse = None  # type: ignore[assignment]
    CallbackHandler = None  # type: ignore[assignment]
    LANGFUSE_AVAILABLE = False

    def observe(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    @contextmanager
    def propagate_attributes(**_kwargs: Any) -> Iterator[None]:
        yield

    def get_client() -> Any:
        raise RuntimeError("Langfuse SDK is not installed")


logger = logging.getLogger(__name__)

_LANGFUSE_INITIALIZED = False
_TRACE_ATTRIBUTES: ContextVar[dict[str, Any] | None] = ContextVar(
    "lesson12_trace_attributes",
    default=None,
)


# Returns a fresh Settings instance for tracing configuration lookups.
# Повертає новий екземпляр Settings для читання конфігурації трасування.
def _settings() -> Settings:
    return Settings()


# Splits a comma-separated tag string into unique trimmed tags.
# Розбиває рядок тегів через кому на унікальні значення без зайвих пробілів.
def _split_tags(raw_tags: str | None) -> list[str]:
    if not raw_tags:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for tag in raw_tags.split(","):
        clean = tag.strip()
        if clean and clean not in seen:
            out.append(clean)
            seen.add(clean)
    return out


# Checks whether Langfuse tracing is available and fully configured.
# Перевіряє, чи Langfuse tracing доступний і повністю налаштований.
def is_langfuse_enabled(settings: Settings | None = None) -> bool:
    settings = settings or _settings()
    return bool(
        LANGFUSE_AVAILABLE
        and settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    )


# Initializes the Langfuse client once per process when credentials are present.
# Ініціалізує клієнт Langfuse один раз на процес, якщо задано облікові дані.
def init_langfuse(settings: Settings | None = None) -> bool:
    global _LANGFUSE_INITIALIZED

    settings = settings or _settings()
    if _LANGFUSE_INITIALIZED:
        return True
    if not is_langfuse_enabled(settings):
        return False

    Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        tracing_enabled=settings.langfuse_enabled,
    )
    _LANGFUSE_INITIALIZED = True
    logger.info("Langfuse tracing initialized base_url=%s", settings.langfuse_base_url)
    return True


# Returns default trace tags from settings as a normalized list.
# Повертає стандартні теги trace з налаштувань як нормалізований список.
def default_trace_tags(settings: Settings | None = None) -> list[str]:
    return _split_tags((settings or _settings()).langfuse_default_tags)


# Builds a default user id for trace attribution in CLI runs.
# Формує стандартний user id для прив'язки trace у CLI-запусках.
def default_user_id() -> str:
    return getpass.getuser() or "cli-user"


# Returns current trace attributes stored in the execution context.
# Повертає поточні атрибути trace, збережені в контексті виконання.
def current_trace_attributes() -> dict[str, Any] | None:
    return _TRACE_ATTRIBUTES.get()


# Opens one root MAS trace context and propagates trace attributes to nested work.
# Відкриває один кореневий MAS trace-контекст і прокидує атрибути в усі вкладені виклики.
@contextmanager
def mas_trace(
    *,
    trace_name: str,
    session_id: str,
    user_id: str,
    tags: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    settings = _settings()
    attrs = {
        "trace_name": trace_name,
        "session_id": session_id,
        "user_id": user_id,
        "tags": tags or default_trace_tags(settings),
        "enabled": init_langfuse(settings),
    }
    token = _TRACE_ATTRIBUTES.set(attrs)
    try:
        with propagate_attributes(
            trace_name=trace_name,
            session_id=session_id,
            user_id=user_id,
            tags=attrs["tags"],
        ):
            yield attrs
    finally:
        _TRACE_ATTRIBUTES.reset(token)


# Builds LangChain invoke config with run name, metadata, and Langfuse callbacks.
# Формує конфіг invoke для LangChain з run name, metadata та callback-ами Langfuse.
def build_langchain_config(
    base_config: dict[str, Any] | None = None,
    *,
    run_name: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = deepcopy(base_config) if base_config else {}
    attrs = current_trace_attributes()

    if run_name and "run_name" not in config:
        config["run_name"] = run_name

    metadata = dict(config.get("metadata") or {})
    if attrs is not None:
        metadata.setdefault("langfuse_session_id", attrs["session_id"])
        metadata.setdefault("langfuse_user_id", attrs["user_id"])
        metadata.setdefault("langfuse_tags", list(attrs["tags"]))
    if extra_metadata:
        metadata.update(extra_metadata)
    if metadata:
        config["metadata"] = metadata

    if attrs is not None and attrs["enabled"] and CallbackHandler is not None:
        callbacks = list(config.get("callbacks") or [])
        callbacks.append(CallbackHandler())
        config["callbacks"] = callbacks

    return config


# Flushes buffered Langfuse events for the current traced execution context.
# Скидає буферизовані події Langfuse для поточного traced-контексту виконання.
def flush_langfuse() -> None:
    attrs = current_trace_attributes()
    if not attrs or not attrs["enabled"]:
        return
    try:
        get_client().flush()
    except Exception:  # pragma: no cover
        logger.exception("Langfuse flush failed")
