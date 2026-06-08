# TASKS — enterprise-rag-pipeline

**Completion:** 90%
**Last Audit:** 2025-01-30
**Repo Role:** NVIDIA SA production RAG reference implementation with CRAG orchestration, multi-backend vector stores, and RAGAS evaluation.

---

## Priority 1 — Critical Gaps (Must Fix)

These issues block production readiness, violate cross-repo standards, or create observability blind spots.

### 1.1 — Add Langfuse Tracing to pipeline/generator.py and Orchestrator

- [ ] **`pipeline/generator.py`** — Add `get_langfuse_handler()` (or import from `core/observability.py`); inject `callbacks=[langfuse_handler]` into every `ChatOpenAI` construction.
  - Acceptance: `grep -n "ChatOpenAI(" pipeline/generator.py` shows `callbacks=` on every match.
- [ ] **`pipeline/generator.py`** — All `chain.ainvoke()` or `llm.ainvoke()` calls include `config={"callbacks": get_callbacks()}`.
  - Acceptance: No LLM invocation in `pipeline/generator.py` lacks a callback config.
- [ ] **`orchestrator/`** (all node files) — All LLM calls in CRAG nodes (grade, generate, rewrite) include Langfuse handler.
  - Acceptance: Each orchestrator node that calls an LLM passes the handler.
- [ ] **`core/observability.py`** (create) — Centralize `get_langfuse_handler()` and `get_callbacks()` here so no duplication across files.
  - Acceptance: File exists; `pipeline/generator.py` and all orchestrator files import from it.
- [ ] **`.env.template`** — Add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
  - Acceptance: All three keys present in `.env.template` with comments.
- [ ] **`requirements.txt`** — Add `langfuse>=2.0.0`.
  - Acceptance: `python -c "import langfuse"` exits 0 after install.
- [ ] **`tests/conftest.py`** — Add `mock_langfuse` and `mock_observability` fixtures.
  - Acceptance: All existing tests pass after Langfuse changes; no new network calls in test suite.

> Use the `.cursor/agents/add-langfuse-tracing.md` agent for this task.

---

### 1.2 — Add Root-Level docker-compose.yml

- [ ] **`docker-compose.yml`** (create at repo root) — Developer convenience compose with `env_file: .env`, build context `.`, `deploy/Dockerfile`.
  - Acceptance: `docker compose config` exits 0; `docker compose up` starts API on port 8000.
- [ ] **`README.md`** — Update "Getting Started" to use `docker compose up` from root.
  - Acceptance: No instruction to `cd deploy/` in the README quick start section.

> Use the `.cursor/agents/fix-docker-compose.md` agent (pattern from nvidia-nim-agent-toolkit) for this task.

---

### 1.3 — Update docs/architecture.md to Document All Directories

- [ ] **`docs/architecture.md`** — Add documentation for `orchestrator/` (CRAG pattern, graph flow, node names).
  - Acceptance: Section "orchestrator/" is present with CRAG flow diagram and state TypedDict description.
- [ ] **`docs/architecture.md`** — Add documentation for `embeddings/` (nim_embedder.py, chunker.py).
  - Acceptance: Section "embeddings/" is present with env var configuration documented.
- [ ] **`docs/architecture.md`** — Add documentation for `vectorstore/` (factory, faiss_store, pgvector_store).
  - Acceptance: Section "vectorstore/" is present with backend selection via `VECTORSTORE_BACKEND` documented.
- [ ] **`docs/architecture.md`** — Add documentation for `retriever/` (hybrid_retriever.py, fusion method).
  - Acceptance: Section "retriever/" is present with description of dense+sparse fusion.
- [ ] **`docs/architecture.md`** — Add complete Environment Variables table.
  - Acceptance: All env vars in `.env.template` appear in the architecture doc table.
- [ ] **`docs/architecture.md`** — Add High-Level Data Flow diagram (ASCII or mermaid).
  - Acceptance: Data flow from document source to API response is visually represented.

> Use the `.cursor/agents/update-architecture-docs.md` agent for this task.

---

## Priority 2 — Polish (Should Fix)

### 2.1 — Add VectorStoreBase ABC

- [ ] **`backends/`** or **`vectorstore/`** — Define `VectorStoreBase` ABC with abstract methods: `add_documents`, `similarity_search`, `delete`. Have `pgvector.py` and `milvus.py` implement it.
  - Acceptance: `isinstance(store, VectorStoreBase)` returns True for all backend instances. mypy validates the implementations.

### 2.2 — Add Custom Exception Classes

- [ ] **`core/exceptions.py`** (create) — Define `IngestionError`, `EmbeddingError`, `RetrievalError`, `GenerationError` as subclasses of `Exception`.
  - Acceptance: File exists; each exception class has a docstring; at least one raise site per exception type exists in the codebase.

### 2.3 — CI Coverage Gate

- [ ] **`.github/workflows/ci.yml`** — Add `--cov-fail-under=80` to pytest command.
  - Acceptance: CI fails if coverage for `pipeline/`, `embeddings/`, `retriever/`, `orchestrator/` drops below 80%.

### 2.4 — RAGAS Baseline Results

- [ ] **`evals/results/`** (create directory) — Run a baseline RAGAS eval against the sample dataset and commit the results JSON as the baseline.
  - Acceptance: `evals/results/baseline_ragas_results.json` exists with all four metrics documented.

> Use the `.cursor/agents/run-ragas-eval.md` agent for this task.

### 2.5 — Chunker Configurable via Environment

- [ ] **`embeddings/chunker.py`** — Verify `CHUNK_SIZE` and `CHUNK_OVERLAP` are read from environment (not hardcoded). Add fallback defaults with logging.
  - Acceptance: Setting `CHUNK_SIZE=256 python -c "from embeddings.chunker import Chunker"` uses 256 as the chunk size.

### 2.6 — FAISS Production Warning

- [ ] **`vectorstore/faiss_store.py`** — Add `logger.warning("Using FAISS in-memory store — data will not persist across restarts. Not suitable for production.")` in the `__init__` method.
  - Acceptance: Starting the app with `VECTORSTORE_BACKEND=faiss` logs this warning.

### 2.7 — Health Check Enhancement

- [ ] **`api/`** — Ensure `GET /health` returns `{"status": "ok", "nim_reachable": bool, "vectorstore_backend": str}`.
  - Acceptance: Endpoint returns these three fields; `nim_reachable` reflects actual NIM ping result.

---

## Priority 3 — Enhancements (Nice to Have)

### 3.1 — Add Async Ingestion Support

- [ ] **`ingestion/`** — Wrap all synchronous loaders with `asyncio.to_thread` at the `pipeline/indexer.py` layer for non-blocking ingestion.
  - Acceptance: `pipeline/indexer.py` uses `asyncio.to_thread` when calling sync loaders; ingestion does not block the event loop.

### 3.2 — Multi-Format Eval Dataset Support

- [ ] **`evals/ragas_runner.py`** — Support both `.json` (list of dicts) and `.jsonl` (newline-delimited) dataset formats.
  - Acceptance: `--dataset` flag accepts both formats without additional flags.

### 3.3 — k8s Helm Chart

- [ ] **`deploy/helm/`** (create) — Basic Helm chart wrapping the k8s manifests in `deploy/k8s/`.
  - Acceptance: `helm lint deploy/helm/` passes. Values file documents all configurable parameters.

### 3.4 — ISV Customization Guide

- [ ] **`docs/isv-customization.md`** (create) — Step-by-step guide for ISV partners: how to swap the vector backend, add a new ingestion loader, change the embedding model, and modify the CRAG graph.
  - Acceptance: Document covers all four customization scenarios with code examples.

### 3.5 — Add Reranker Support

- [ ] **`retriever/hybrid_retriever.py`** — Add optional reranking step after hybrid retrieval using a NIM reranker model (controlled by `USE_RERANKER=true` env var).
  - Acceptance: When `USE_RERANKER=true`, retrieved chunks are reranked before being passed to the generator. Feature is documented in architecture.md.

---

## Cross-Repo Tasks

These tasks apply identically across `nvidia-nim-agent-toolkit`, `enterprise-rag-pipeline`, and `multi-agent-reference-architecture`.

- [ ] **All repos** — Confirm `langfuse>=2.0.0` is in each repo's `requirements.txt`.
- [ ] **All repos** — Confirm `docker-compose.yml` exists at each repo root.
- [ ] **All repos** — Confirm `.env.template` documents `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
- [ ] **All repos** — Confirm `ruff check . && mypy .` passes clean in each repo independently.
- [ ] **All repos** — Confirm LangGraph state is `TypedDict` (not `dict`) in every graph file.
- [ ] **All repos** — Confirm no `from langchain.` imports (must be `langchain_core`, `langchain_openai`, or `langchain_community`).
