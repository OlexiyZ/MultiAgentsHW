from __future__ import annotations

import json
from pathlib import Path

from langfuse import Langfuse

from config import Settings


PROMPTS_FILE = Path(__file__).resolve().parent / "langfuse_prompts" / "system_prompts.json"


def main() -> None:
    """Creates prompt versions in Langfuse from the local prompt manifest.
    Створює версії промптів у Langfuse з локального manifest-файлу."""

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

    for prompt in prompts:
        langfuse.create_prompt(
            name=prompt["name"],
            type=prompt["type"],
            prompt=prompt["prompt"],
            labels=prompt["labels"],
        )
        print(f"Uploaded prompt: {prompt['name']} labels={prompt['labels']}")

    try:
        langfuse.flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
