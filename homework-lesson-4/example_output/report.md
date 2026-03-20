# Example Research Report

## Topic
Comparison of naive RAG and sentence-window retrieval.

## Key Findings
- Naive RAG is simpler to implement, but it often loses local context when chunks are too coarse or too small.
- Sentence-window retrieval improves answer quality when precise neighboring context matters.
- The better approach depends on the document structure, latency budget, and indexing complexity.

## Sources
- https://python.langchain.com/
- https://docs.llamaindex.ai/
