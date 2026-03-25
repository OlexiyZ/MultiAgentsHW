PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv ingest-llama ingest-langchain run-rag run-rag-debug test_lesson_5

help:
	@echo "Targets:"
	@echo "  make venv              - create .venv and install requirements"
	@echo "  make ingest-llama      - ingest data/ PDFs via LlamaIndex -> index/"
	@echo "  make ingest-langchain  - ingest data/ PDFs via LangChain -> index/"
	@echo "  make run-rag           - interactive agent"
	@echo "  make run-rag-debug     - agent with --debug"
	@echo "  make test_lesson_5     - pytest"

venv: $(PYTHON)

$(PYTHON):
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

ingest-llama: $(PYTHON)
	$(PYTHON) ingest_llama_flavour.py

ingest-langchain: $(PYTHON)
	$(PYTHON) ingest_langchain_flavour.py

run-rag: $(PYTHON)
	$(PYTHON) main.py

run-rag-debug: $(PYTHON)
	$(PYTHON) main.py --debug

test_lesson_5: $(PYTHON)
	$(PYTHON) -m pytest tests/ -q
