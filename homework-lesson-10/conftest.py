"""
Root conftest.py for homework-lesson-10 test suite.

Adds homework-lesson-8 to sys.path so that agents, schemas, tools, etc.
can be imported without installation, and loads the lesson-8 .env file so
that API keys (OPENAI_API_KEY, etc.) are available to any test that needs them.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# DeepEval/Rich prints unicode status icons; Windows cp1252 consoles can fail
# during test output rendering unless streams are configured for UTF-8.
# DeepEval/Rich виводить unicode-іконки статусу; Windows-консолі cp1252 можуть падати
# під час рендерингу тестового виводу, якщо потоки не налаштовані на UTF-8.
for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name)
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

# Resolve lesson-8 directory relative to this conftest
# Визначаємо директорію lesson-8 відносно цього conftest
LESSON8_DIR = Path(__file__).parent.parent / "homework-lesson-8"

# Make lesson-8 importable (agents, schemas, tools, retriever, …)
# Додаємо lesson-8 до import path (agents, schemas, tools, retriever, ...)
sys.path.insert(0, str(LESSON8_DIR))

# Load API keys and other settings from lesson-8's .env
# Завантажуємо API-ключі та інші налаштування з lesson-8 .env
load_dotenv(LESSON8_DIR / ".env")
