# Research Report

## Topic
Comparison of naive RAG, sentence-window retrieval, and parent-child retrieval

## Key Findings
- Naive RAG is the simplest to implement and works well for small corpora, but it often loses local context because chunks are retrieved independently.
- Sentence-window retrieval improves answer grounding by attaching neighboring sentences to the retrieved chunk, which is useful when meaning depends on nearby context.
- Parent-child retrieval improves recall and readability by indexing small child chunks while returning larger parent chunks to the model.

## Trade-offs
| Approach | Strengths | Weaknesses | Best Fit |
| --- | --- | --- | --- |
| Naive RAG | Fast to build, low operational complexity | Weak contextual continuity, more irrelevant chunks | Prototypes and small datasets |
| Sentence-window | Better local context, less fragmentation | More tuning around window size | Knowledge bases where neighboring text matters |
| Parent-child | Good balance between precise retrieval and rich context | More indexing complexity | Production systems with larger documents |

## Suggested Conclusion
Start with naive RAG for baseline evaluation. Move to sentence-window retrieval when local context loss hurts answer quality, and adopt parent-child retrieval when you need stronger retrieval precision without sacrificing the context returned to the model.

## Sources
- https://docs.llamaindex.ai/
- https://python.langchain.com/
