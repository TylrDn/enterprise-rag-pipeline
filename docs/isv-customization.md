# ISV Customization Guide

This guide shows how ISV partners customize the enterprise RAG pipeline without forking core orchestration logic. Most changes are configuration-driven; loader and backend swaps require small, isolated Python modules.

## 1. Swap the Vector Backend

Set the backend via environment variable — no code changes required:

```bash
# Postgres + pgvector (default, production)
VECTORSTORE_BACKEND=pgvector
PGVECTOR_URL=postgresql+psycopg://user:pass@host:5432/ragdb

# Milvus (enterprise scale)
VECTORSTORE_BACKEND=milvus
MILVUS_URI=http://milvus:19530
MILVUS_COLLECTION=my_docs

# FAISS (local dev only — in-memory, not persisted)
VECTORSTORE_BACKEND=faiss
```

The factory in `vectorstore/factory.py` resolves the backend at startup. All pipeline code uses `get_vector_store(embedder)`.

To add a custom backend:

1. Implement `VectorStoreBase` in a new module under `backends/`.
2. Register it in `vectorstore/factory.py`.
3. Add env vars to `.env.template` and document them in `docs/architecture.md`.

## 2. Add a New Ingestion Loader

1. Create `ingestion/my_loader.py` with a class exposing `load(...) -> list[Document]`.
2. Export it from `ingestion/__init__.py`.
3. Index via the API or pipeline:

```python
from ingestion.my_loader import MyLoader
from orchestrator.pipeline import RAGPipeline

pipeline = RAGPipeline()
docs = MyLoader().load("source://path")
pipeline.index(docs)
```

For async ingestion from the API layer, wrap sync loaders with `asyncio.to_thread` (see `ingestion.ingest_pdf` / `ingestion.ingest_url`).

## 3. Change the Embedding Model

Update `.env`:

```bash
NIM_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_API_KEY=your-key
```

The embedder in `pipeline/embedder.py` reads `NIM_EMBEDDING_MODEL` at startup. Re-index documents after changing models — vector dimensions must match the store schema.

Adjust chunking independently:

```bash
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

## 4. Modify the CRAG Graph

The graph topology lives in `orchestrator/graph.py`. To add a node:

1. Implement `async def my_node(state: RAGState) -> dict` in `orchestrator/nodes/`.
2. Register it in `build_rag_graph()` with edges and optional conditional routers.
3. Extend `RAGState` in `orchestrator/state.py` if new fields are needed.
4. Add tests in `tests/test_graph.py`.

Example — insert a query-rewrite node before retrieval:

```python
graph.add_node("rewrite", rewrite_query)
graph.set_entry_point("rewrite")
graph.add_edge("rewrite", "retrieve")
```

Keep LLM calls routed through `orchestrator/nodes/_llm.py` so Langfuse tracing stays consistent.

## 5. Tune Retrieval and Reranking

```bash
RETRIEVER_TOP_K=5
BM25_WEIGHT=0.4
DENSE_WEIGHT=0.6
USE_RERANKER=true
RERANK_BACKEND=nim          # or cross-encoder
NIM_RERANK_MODEL=nvidia/nv-rerankqa-mistral-4b-v3
```

## 6. Evaluation Workflow

Run baseline RAGAS evals after customization:

```bash
python -m evals.ragas_runner \
  --dataset evals/datasets/sample_qa.json \
  --output-file evals/results/my_run.json \
  --metrics faithfulness answer_relevancy
```

Compare against `evals/results/baseline_ragas_results.json`.

## 7. Deployment

From the repo root:

```bash
cp .env.template .env
docker compose up --build
curl http://localhost:8000/health
```

For Kubernetes, use `deploy/helm/` or the raw manifests in `deploy/k8s/`.
