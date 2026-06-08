# Enterprise RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system built on **NVIDIA NIM**, **pgvector**, and **LangChain**.

## Architecture

```
Documents
    │
    ▼
[ Indexer ] ── RecursiveCharacterTextSplitter ── SHA-256 dedup ── batch upsert
    │
    ▼
[ PgVectorBackend ]  (cosine, JSONB metadata, langchain-postgres)
    │
    ▼
[ HybridRetriever ]
    ├── BM25Retriever (sparse / lexical)
    └── Dense VectorStoreRetriever (semantic)
           └── EnsembleRetriever (RRF, alpha=0.7)
                  └── CrossEncoderReranker (ms-marco-MiniLM-L-6-v2)
    │
    ▼
[ Generator ]  ── NVIDIA NIM (meta/llama3-70b-instruct)
    ├── Grounding guard (refuses on empty context)
    └── Hallucination check (secondary LLM grading call)
    │
    ▼
[ FastAPI ]  POST /ingest  |  POST /query  |  GET /query/stream  |  GET /health
```

## Quickstart

### 1. Clone & configure
```bash
git clone https://github.com/TylrDn/enterprise-rag-pipeline.git
cd enterprise-rag-pipeline
cp .env.template .env
# Fill in NVIDIA_API_KEY and PGVECTOR_URL in .env
```

### 2. Run with Docker Compose
```bash
cd deploy
docker compose up --build
```
The API will be available at `http://localhost:8080`. Interactive docs at `/docs`.

### 3. Local development
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.server:app --reload --port 8080
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Chunk, embed, and store documents |
| `POST` | `/query` | Hybrid retrieve + grounded generation |
| `GET` | `/query/stream` | SSE streaming answer |
| `GET` | `/health` | Liveness + backend readiness |

### Example: Ingest
```bash
curl -X POST http://localhost:8080/ingest \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["NVIDIA NIM provides optimised inference."], "metadatas": [{"source": "docs.pdf"}]}'
```

### Example: Query
```bash
curl -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does NVIDIA NIM do?"}'
```

## Configuration

| File | Purpose |
|------|---------|
| `configs/rag.yaml` | Embedding model, chat model, retriever settings, grader thresholds |
| `configs/pipeline.yaml` | Chunk size, overlap, batch size |
| `configs/backends.yaml` | pgvector and Milvus connection config |
| `.env` | Secrets: `NVIDIA_API_KEY`, `PGVECTOR_URL`, `LANGFUSE_*` |

## Project Structure

```
enterprise-rag-pipeline/
├── api/               FastAPI server
├── backends/          Vector store backends (pgvector)
├── configs/           YAML configuration
├── deploy/            Dockerfile + docker-compose
├── evals/             RAGAS evaluation harness
├── orchestrator/      End-to-end RAGPipeline wrapper
├── pipeline/          Core: embedder, indexer, retriever, generator
├── tests/             Pytest suite (mocked, no live services needed)
└── .github/workflows/ CI (pytest + coverage on Python 3.11 & 3.12)
```

## Running Tests

```bash
pytest tests/ -v --cov=pipeline --cov=backends --cov=api --cov=orchestrator
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NVIDIA_API_KEY` | Yes | NVIDIA NIM API key |
| `PGVECTOR_URL` | Yes | SQLAlchemy async connection string |
| `EMBEDDING_BACKEND` | No | `nim` (default) or `huggingface` |
| `RERANKER_MODEL` | No | HuggingFace cross-encoder model name |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse observability |
| `LANGFUSE_SECRET_KEY` | No | Langfuse observability |
