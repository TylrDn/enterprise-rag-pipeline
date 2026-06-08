# Architecture — enterprise-rag-pipeline

## System Overview
Multi-source RAG pipeline with corrective retrieval loop, hallucination grounding, and RAGAS evaluation.

```mermaid
graph TD
    PDF[PDF Files] --> ING[Ingestion Layer]
    SQL[SQL Tables] --> ING
    WEB[Web URLs] --> ING
    SLACK[Slack Channels] --> ING
    ING --> CHUNK[Chunker]
    CHUNK --> NIM_EMB[NIM Embedder\nnvidia/nv-embedqa-e5-v5]
    NIM_EMB --> PG[(pgvector\nPostgres)]
    NIM_EMB --> FAISS[(FAISS\nIn-Memory)]

    USER[User Question] --> API[FastAPI :8000\nPOST /query]
    API --> RW[query_rewriter node\nNIM LLM]
    RW --> RET[retriever node\nHybrid Dense+BM25]
    RET --> PG
    RET --> FAISS
    RET --> GRADE[grader node\nrelevance filter]
    GRADE -->|no relevant docs| RW
    GRADE -->|docs pass| GEN[generator node\nNIM LLM + context]
    GEN --> HALL[hallucination_checker\nfaithfulness score]
    HALL -->|score >= 0.6| OUT[Answer + Sources]
    HALL -->|score < 0.6| RW
    OUT --> API

    API --> LS[LangSmith Tracing]
    RAGAS[RAGAS Eval Runner] --> PG
    RAGAS --> LS
```

## Corrective RAG Loop
| Condition | Action |
|---|---|
| No relevant docs after grading | Rewrite query → re-retrieve |
| Hallucination score < 0.6 | Rewrite query → re-retrieve |
| retry_count >= 2 | Force generate with best available |

## Ingestion Sources
| Source | Loader | Chunking |
|---|---|---|
| PDF | pdfplumber / unstructured | recursive 512/64 |
| SQL | SQLAlchemy row→doc | none (row = chunk) |
| Web | httpx + BeautifulSoup | recursive 512/64 |
| Slack | slack-sdk messages | recursive 512/64 |

## Eval Stack
- **RAGAS** — faithfulness, answer relevancy, context precision, context recall
- **LangSmith** — experiment tracking, dataset versioning, regression CI
