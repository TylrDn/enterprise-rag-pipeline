---
name: update-architecture-docs
description: Invoke this agent when docs/architecture.md is missing documentation for one or more top-level directories — specifically when orchestrator/, embeddings/, vectorstore/, or retriever/ are not described in the architecture document.
model: inherit
readonly: false
---

# Update Architecture Documentation — enterprise-rag-pipeline

## Objective

Rewrite `docs/architecture.md` so that it accurately and completely documents every top-level directory in the repository. The current version omits documentation for `orchestrator/`, `embeddings/`, `vectorstore/`, and `retriever/`. The updated document must serve as the canonical architectural reference for ISV partners, new team members, and the Solutions Architect portfolio.

## Context

`enterprise-rag-pipeline` has grown to include sophisticated components beyond the initial ingestion and pipeline modules. The architecture doc predates the addition of the CRAG orchestrator, the NIM embedder, the vector store abstraction layer, and the hybrid retriever. Any ISV or engineer reading only the docs would not know these exist, which undermines the repo's value as a reference implementation.

## Files to Touch

1. `docs/architecture.md` — Primary file. Rewrite to be comprehensive.
2. `README.md` — Optionally update the "Architecture" section to reference `docs/architecture.md` and confirm the link is accurate.

## Step-by-Step Instructions

### Step 1 — Inventory the Repo

Before writing, read every top-level directory and its key files:

- `ingestion/` → list all loader files
- `pipeline/` → list `embedder.py`, `indexer.py`, `retriever.py`, `generator.py`
- `backends/` → list `pgvector.py`, `milvus.py`
- `evals/` → list `ragas_eval.py`, `langsmith_eval.py`, `ragas_runner.py`, `datasets/`
- `vectorstore/` → list `faiss_store.py`, `pgvector_store.py`, and factory module
- `embeddings/` → list `nim_embedder.py`, `chunker.py`
- `orchestrator/` → list graph file and node files
- `retriever/` → list `hybrid_retriever.py`
- `deploy/` → list `Dockerfile`, `docker-compose.yml`, `k8s/` contents
- `api/` → list server and route files
- `tests/` → note structure
- `notebooks/` → list notebooks

Use this inventory to produce accurate documentation.

### Step 2 — Rewrite `docs/architecture.md`

The updated document must include all sections below. Write each section with enough detail that a new engineer or ISV partner can understand what lives there and why.

---

Use this template for `docs/architecture.md`:

```markdown
# Architecture — enterprise-rag-pipeline

## Overview

`enterprise-rag-pipeline` is the NVIDIA SA production-grade Retrieval-Augmented Generation reference implementation. It provides a complete document ingestion, embedding, storage, retrieval, and generation pipeline, orchestrated by a Corrective RAG (CRAG) LangGraph graph. It is designed to be deployed against NVIDIA NIM inference endpoints and supports multiple vector store backends.

## High-Level Data Flow

```
Documents (PDF / Slack / Web / SQL)
         │
         ▼
  ingestion/ loaders
         │
         ▼
  embeddings/chunker.py  ──→  embeddings/nim_embedder.py (NIM Embeddings API)
         │
         ▼
  vectorstore/ factory  ──→  backends/ (pgvector | milvus | faiss)
         │
         ▼
  retriever/hybrid_retriever.py (dense + sparse fusion)
         │
         ▼
  orchestrator/ CRAG LangGraph
    ├── retrieve_documents
    ├── grade_documents  ──→  (relevant?) YES → generate_answer
    │                                      NO  → web_search_fallback → generate_answer
    └── generate_answer  ──→  API response
```

## Directory Reference

### `ingestion/`
Document loaders that transform raw sources into `langchain_core.documents.Document` objects.

| File | Source | Library |
|---|---|---|
| `pdf_loader.py` | PDF files | pypdf2 |
| `slack_loader.py` | Slack channels/exports | slack-sdk |
| `web_loader.py` | Web URLs | beautifulsoup4, requests |
| `sql_loader.py` | SQL databases | sqlalchemy |

All loaders implement a `load() -> list[Document]` interface and preserve source metadata on each document.

### `embeddings/`
NIM embedding and text chunking utilities.

- **`nim_embedder.py`** — Wraps `langchain_openai.OpenAIEmbeddings` pointed at the NIM embedding endpoint (`NIM_BASE_URL`). Reads `NIM_EMBEDDING_MODEL` from environment. Exposes `embed_documents()` and `embed_query()` with tenacity retry logic.
- **`chunker.py`** — Uses `RecursiveCharacterTextSplitter` with configurable chunk size (`CHUNK_SIZE` env var, default 512) and overlap (`CHUNK_OVERLAP` env var, default 50). Preserves document metadata on all output chunks.

### `pipeline/`
High-level pipeline orchestration layer that wires ingestion, embedding, indexing, retrieval, and generation.

- **`embedder.py`** — Drives the embedding step: takes `list[Document]`, chunks via `chunker.py`, embeds via `nim_embedder.py`.
- **`indexer.py`** — Drives the indexing step: takes embedded chunks, writes to the configured vector store backend.
- **`retriever.py`** — Pipeline-layer retrieval wrapper. Delegates to `retriever/hybrid_retriever.py`.
- **`generator.py`** — RAG generation step. Constructs a LangChain RAG chain with `ChatOpenAI` (NIM), applies Langfuse tracing, and produces the final answer.

### `vectorstore/`
Vector store factory and per-backend wrappers.

- **`faiss_store.py`** — In-memory FAISS vector store. Development and offline use only. Logs a warning when active.
- **`pgvector_store.py`** — pgvector-backed store using `langchain_community.vectorstores.PGVector`. Reads `PGVECTOR_URL`, `PGVECTOR_COLLECTION` from environment.
- **Factory module** — Reads `VECTORSTORE_BACKEND` env var and returns the appropriate store instance. Pipeline code imports only from the factory.

### `backends/`
Low-level database adapters implementing the `VectorStoreBase` ABC.

- **`pgvector.py`** — psycopg2 + pgvector adapter for direct SQL operations (schema management, bulk insert).
- **`milvus.py`** — pymilvus adapter. Reads `MILVUS_URI`, `MILVUS_COLLECTION` from environment.

### `retriever/`
Hybrid retrieval combining dense (vector similarity) and sparse (keyword/BM25) search.

- **`hybrid_retriever.py`** — Accepts a query, runs dense retrieval against the configured vector store and sparse retrieval, then fuses scores using Reciprocal Rank Fusion (RRF). Configurable `top_k` (default 5).

### `orchestrator/`
CRAG (Corrective RAG) LangGraph orchestrator.

The CRAG pattern extends standard RAG by grading retrieved documents for relevance. If documents are insufficient, a web search fallback is triggered before generation.

**Graph flow:**
```
retrieve_documents → grade_documents → [generate_answer | web_search_fallback → generate_answer]
```

- **State** — `RAGState` TypedDict defined in the state module. Tracks question, documents, generation, grade scores, iteration count.
- **Nodes** — `retrieve_documents`, `grade_documents`, `web_search_fallback`, `generate_answer`.
- **Conditional edge** — `route_generation` router inspects grade scores and routes to generate or web search.
- **Max iterations** — Capped at 3 (configurable via `CRAG_MAX_ITERATIONS` env var).

### `evals/`
Offline evaluation scripts. Never imported by the main pipeline.

- **`ragas_eval.py`** — Runs RAGAS metrics (faithfulness, answer relevancy, context precision, context recall) on a dataset.
- **`langsmith_eval.py`** — Uploads eval results to LangSmith for tracking across pipeline versions.
- **`ragas_runner.py`** — CLI runner: `python ragas_runner.py --dataset datasets/qa_pairs.json --output-file results.json --metrics faithfulness,answer_relevancy`.
- **`datasets/`** — JSON/JSONL evaluation datasets (question, context, expected answer triples).

### `deploy/`
Deployment artifacts.

- **`Dockerfile`** — Multi-stage Python 3.11 image. Build from repo root: `docker build -f deploy/Dockerfile .`
- **`docker-compose.yml`** — Production compose with database services (PostgreSQL + pgvector, Milvus).
- **`k8s/`** — Kubernetes manifests: Deployment, Service, ConfigMap, PersistentVolumeClaim.

### `api/`
FastAPI server exposing the RAG pipeline as an HTTP API.

Key endpoints:
- `POST /ingest` — Trigger document ingestion from a source.
- `POST /query` — Submit a query and receive a grounded answer.
- `GET /health` — Health check (returns NIM reachability status).

### `tests/`
pytest test suite mirroring the source directory structure.

- Unit tests mock all external calls (NIM API, databases, Langfuse).
- Integration tests in `tests/integration/` require live services and are marked `@pytest.mark.integration`.

### `notebooks/`
Exploratory Jupyter notebooks for development and demonstration. Not imported by the application.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `NIM_BASE_URL` | Yes | NIM inference endpoint base URL |
| `NIM_API_KEY` | Yes | NIM API key |
| `NIM_EMBEDDING_MODEL` | Yes | NIM embedding model name |
| `NIM_GENERATOR_MODEL` | Yes | NIM generator model name |
| `VECTORSTORE_BACKEND` | Yes | `pgvector`, `milvus`, or `faiss` |
| `PGVECTOR_URL` | If pgvector | PostgreSQL connection string |
| `PGVECTOR_COLLECTION` | If pgvector | pgvector collection name |
| `MILVUS_URI` | If milvus | Milvus connection URI |
| `MILVUS_COLLECTION` | If milvus | Milvus collection name |
| `LANGFUSE_PUBLIC_KEY` | Recommended | Langfuse tracing public key |
| `LANGFUSE_SECRET_KEY` | Recommended | Langfuse tracing secret key |
| `LANGFUSE_HOST` | Recommended | Langfuse host (default: cloud) |
| `LANGCHAIN_TRACING_V2` | Optional | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | If LangSmith | LangSmith API key |
| `CHUNK_SIZE` | Optional | Text chunk size in tokens (default: 512) |
| `CHUNK_OVERLAP` | Optional | Chunk overlap in tokens (default: 50) |
| `CRAG_MAX_ITERATIONS` | Optional | CRAG loop max iterations (default: 3) |

## Related Repositories

- [`nvidia-nim-agent-toolkit`](../nvidia-nim-agent-toolkit) — Upstream NIM client and LangGraph patterns.
- [`multi-agent-reference-architecture`](../multi-agent-reference-architecture) — Uses this repo as the RAG backend in multi-agent pipelines.
```

### Step 3 — Update README.md Architecture Section

Find the "Architecture" or "How It Works" section in `README.md`. Replace it with:

```markdown
## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the complete architecture reference, including all pipeline components, directory descriptions, and environment variable documentation.

**Quick summary:** Documents → ingestion loaders → NIM embedder + chunker → vector store (pgvector/Milvus/FAISS) → hybrid retriever → CRAG LangGraph orchestrator → RAG generator → API response.
```

## Acceptance Criteria

- [ ] `docs/architecture.md` documents every top-level directory: `ingestion/`, `pipeline/`, `backends/`, `evals/`, `vectorstore/`, `embeddings/`, `orchestrator/`, `retriever/`, `deploy/`, `api/`, `tests/`, `notebooks/`.
- [ ] `docs/architecture.md` includes a High-Level Data Flow diagram.
- [ ] `docs/architecture.md` includes a complete Environment Variables table.
- [ ] `docs/architecture.md` includes a CRAG graph flow description with node names.
- [ ] `README.md` architecture section links to `docs/architecture.md`.
- [ ] Every directory in the repo root has a corresponding entry in `docs/architecture.md`.
- [ ] Document reads clearly for an ISV partner with no prior context.
- [ ] No placeholder text (`TODO`, `TBD`, `...`) remains in the document.
