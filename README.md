# enterprise-rag-pipeline

[![CI](https://github.com/TylrDn/enterprise-rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/TylrDn/enterprise-rag-pipeline/actions/workflows/ci.yml)

Production-grade multi-source RAG pipeline with **pgvector** and **Milvus** vector backends, **hybrid BM25 + dense retrieval**, cross-encoder or NIM reranking, **Corrective RAG (CRAG)** orchestration, and **RAGAS + LangSmith** evaluation harnesses — built on NVIDIA NIM embeddings and generation.

## Architecture

```
User Query
    ↓
FastAPI /query (CRAG LangGraph)
    ↓
HybridRetriever (BM25 + Dense + Rerank)
    ↓
[pgvector | Milvus | FAISS]
    ↓
NVIDIA NIM (generation + embeddings)
```

See [docs/architecture.md](docs/architecture.md) for the full Mermaid diagram and [docs/isv-customization.md](docs/isv-customization.md) for ISV partner guidance.

## Quickstart

```bash
cp .env.template .env
# Add NVIDIA_API_KEY to .env

pip install -r requirements.txt

# Start Postgres + API from repo root
docker compose up --build -d

# Or run locally against an existing Postgres
uvicorn api.server:app --reload --port 8000
curl http://localhost:8000/health
```

## Docker

```bash
docker compose up --build
```

Production-parameterized compose lives in `deploy/docker-compose.yml`.

## Ingestion

```python
from ingestion.pdf_loader import PDFLoader
from pipeline.embedder import get_embedder
from pipeline.indexer import Indexer
from vectorstore.factory import get_vector_store

loader = PDFLoader()
docs = loader.load_directory("./data/pdfs")

embedder = get_embedder()
backend = get_vector_store(embedder.as_langchain())
indexer = Indexer(backend=backend)
indexer.index(docs)
```

Async helpers and API routes:

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["Hello world"], "metadatas": [{"source": "demo"}]}'
```

## Retrieval & Generation

```python
from pipeline.retriever import HybridRetriever
from pipeline.generator import RAGGenerator

retriever = HybridRetriever(
    vector_store=backend,
    corpus_documents=docs,
)
generator = RAGGenerator(retriever.as_langchain_retriever())
result = generator.generate_with_sources("What is the refund policy?")
print(result["answer"])
print(result["sources"])
```

CRAG orchestration is the default `/query` path (`use_crag: true`).

## Evaluation

```bash
python -m evals.ragas_runner --output-file evals/results/ragas_results.json
python -m evals.langsmith_eval
```

Baseline metrics: `evals/results/baseline_ragas_results.json`.

## Key Components

| Module | Description |
|---|---|
| `ingestion/pdf_loader.py` | PyMuPDF + RecursiveCharacterTextSplitter |
| `ingestion/sql_loader.py` | Schema-aware SQL table → Document |
| `ingestion/web_loader.py` | httpx + BeautifulSoup scraper |
| `ingestion/slack_loader.py` | Slack export JSON → Documents |
| `pipeline/embedder.py` | NIM embedding wrapper with retries |
| `pipeline/retriever.py` | BM25 + dense hybrid + optional rerank |
| `pipeline/generator.py` | NIM-backed RAG generation chain |
| `orchestrator/graph.py` | CRAG LangGraph orchestration |
| `backends/pgvector_backend.py` | Postgres pgvector store |
| `backends/milvus_backend.py` | Milvus vector store |
| `evals/ragas_runner.py` | RAGAS CLI with `.json` / `.jsonl` datasets |
| `evals/langsmith_eval.py` | LangSmith dataset + evaluator harness |

## Environment Variables

| Variable | Description |
|---|---|
| `NVIDIA_API_KEY` | NVIDIA NIM API key |
| `NIM_BASE_URL` | NIM endpoint (default: integrate.api.nvidia.com) |
| `VECTORSTORE_BACKEND` | `pgvector`, `milvus`, or `faiss` |
| `PGVECTOR_URL` | SQLAlchemy connection string |
| `LANGFUSE_PUBLIC_KEY` | Langfuse tracing (optional) |
| `LANGSMITH_API_KEY` | LangSmith tracing + evals |

See `.env.template` for the full list.

## Cross-Repo Integration

- [`nvidia-nim-agent-toolkit`](https://github.com/TylrDn/nvidia-nim-agent-toolkit) — DocAgent uses this pipeline as its RAG backend
- [`agentic-guardrails-eval`](https://github.com/TylrDn/agentic-guardrails-eval) — safety eval suite can target this pipeline's `/query` endpoint

## Topics

`rag` `retrieval-augmented-generation` `langchain` `pgvector` `milvus` `ragas` `langsmith` `nemo-retriever` `python` `enterprise-ai`
