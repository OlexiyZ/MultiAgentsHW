from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from langfuse import Langfuse
except ImportError:  # pragma: no cover
    Langfuse = None  # type: ignore[assignment]

from config import Settings


PROMPTS_FILE = Path(__file__).resolve().parent / "langfuse_prompts" / "system_prompts.json"


def _prompt_text(prompt: object) -> str:
    """Returns text from a Langfuse prompt object across SDK versions."""

    compile_method = getattr(prompt, "compile", None)
    if callable(compile_method):
        compiled = compile_method()
        if isinstance(compiled, str):
            return compiled

    raw = getattr(prompt, "prompt", None)
    if isinstance(raw, str):
        return raw

    raise TypeError(
        f"Unsupported Langfuse prompt object: expected text prompt, got {type(prompt).__name__}"
    )


def _current_prompt_text(langfuse: Any, name: str, label: str) -> str | None:
    """Loads current prompt text from Langfuse or returns None when it is absent."""

    try:
        prompt = langfuse.get_prompt(name, label=label)
    except Exception as exc:
        print(f"Prompt not found or not readable: {name} label={label} ({exc})")
        return None
    return _prompt_text(prompt)


def _primary_label(labels: list[str]) -> str:
    return labels[0] if labels else "production"


def main() -> None:
    """Creates Langfuse prompt versions only when local prompt text changed."""

    if Langfuse is None:
        raise RuntimeError(
            "Langfuse SDK is required to sync prompts. Install the 'langfuse' package."
        )

    settings = Settings()
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value()
        if settings.langfuse_public_key
        else None,
        secret_key=settings.langfuse_secret_key.get_secret_value()
        if settings.langfuse_secret_key
        else None,
        base_url=settings.langfuse_base_url,
    )
    prompts = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))

    uploaded = 0
    skipped = 0
    for prompt in prompts:
        name = prompt["name"]
        labels = list(prompt["labels"])
        label = _primary_label(labels)
        local_text = prompt["prompt"]
        current_text = _current_prompt_text(langfuse, name, label)

        if current_text == local_text:
            skipped += 1
            print(f"Skipped unchanged prompt: {name} label={label}")
            continue

        langfuse.create_prompt(
            name=name,
            type=prompt["type"],
            prompt=local_text,
            labels=labels,
        )
        uploaded += 1
        print(f"Uploaded prompt: {name} labels={labels}")

    try:
        langfuse.flush()
    except Exception:
        pass

    print(f"Done. uploaded={uploaded} skipped={skipped}")


if __name__ == "__main__":
    main()
