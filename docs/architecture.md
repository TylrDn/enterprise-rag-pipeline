# Enterprise RAG Pipeline — Architecture

## System Overview

Production-grade multi-source RAG pipeline with pluggable vector backends, hybrid retrieval, and RAGAS-driven evaluation.

## Component Diagram

```mermaid
graph TD
    User([User Query]) --> API[FastAPI /query]
    API --> Gen[RAGGenerator]

    subgraph Retrieval
        Gen --> Hybrid[HybridRetriever]
        Hybrid --> BM25[BM25 Sparse]
        Hybrid --> Dense[Dense Vector]
        BM25 --> Rerank[Cross-Encoder Reranker]
        Dense --> Rerank
    end

    subgraph Vector Backends
        Dense --> PG[pgvector / Postgres]
        Dense --> Milvus[Milvus]
    end

    subgraph Ingestion
        PDF[PDFLoader] --> Indexer
        SQL[SQLLoader] --> Indexer
        Web[WebLoader] --> Indexer
        Slack[SlackLoader] --> Indexer
        Indexer --> PG
        Indexer --> Milvus
    end

    subgraph NIM Layer
        Gen --> NIM[NVIDIA NIM\nOpenAI-compatible]
        Embed[Embedder] --> NIM
    end

    subgraph Eval
        Gen --> RAGAS[ragas_eval.py]
        Gen --> LangSmith[langsmith_eval.py]
    end
```

## Key Design Decisions

- **Backend swappability:** `VECTOR_BACKEND=pgvector|milvus` env var controls which store is used — no code changes
- **Hybrid retrieval:** BM25 (40%) + dense (60%) ensemble, reranked by a cross-encoder for precision
- **NIM embeddings:** `nvidia/nv-embedqa-e5-v5` via the OpenAI-compatible NIM API
- **RAGAS metrics:** faithfulness, answer relevancy, context recall, context precision — tracked per pipeline config

## Integration with nvidia-nim-agent-toolkit

The DocAgent in `nvidia-nim-agent-toolkit` can delegate retrieval to this pipeline's `/query` endpoint, making this repo the RAG backend for the full agent system.
