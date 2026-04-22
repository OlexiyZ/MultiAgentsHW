"""
Root conftest.py for homework-lesson-10 test suite.

Adds homework-lesson-10 to sys.path so that agents, schemas, tools, etc.
can be imported without installation, and loads the lesson-10 .env file so
that API keys (OPENAI_API_KEY, etc.) are available to any test that needs them.

Кореневий conftest.py для тестового набору homework-lesson-10.

Додає homework-lesson-10 до sys.path, щоб agents, schemas, tools тощо можна
було імпортувати без встановлення, і завантажує .env файл lesson-10, щоб
API-ключі (OPENAI_API_KEY тощо) були доступні тестам, які їх потребують.
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

# Resolve the current lesson-10 directory relative to this conftest.
# Визначаємо поточну директорію lesson-10 відносно цього conftest.
LESSON10_DIR = Path(__file__).resolve().parent

# Make lesson-10 importable (agents, schemas, tools, retriever, ...).
# Додаємо lesson-10 до import path (agents, schemas, tools, retriever, ...).
sys.path.insert(0, str(LESSON10_DIR))

# Load API keys and other settings from lesson-10's .env.
# Завантажуємо API-ключі та інші налаштування з lesson-10 .env.
load_dotenv(LESSON10_DIR / ".env")
