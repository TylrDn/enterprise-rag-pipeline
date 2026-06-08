# enterprise-rag-pipeline

[![CI](https://github.com/TylrDn/enterprise-rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/TylrDn/enterprise-rag-pipeline/actions/workflows/ci.yml)

Production-grade multi-source RAG pipeline with **pgvector** and **Milvus** vector backends, **hybrid BM25 + dense retrieval**, cross-encoder reranking, and **RAGAS + LangSmith** evaluation harnesses — built on NVIDIA NIM embeddings and generation.

## Architecture

```
User Query
    ↓
FastAPI /query
    ↓
RAGGenerator
    ↓
HybridRetriever (BM25 + Dense + Rerank)
    ↓
[pgvector | Milvus]
    ↓
NVIDIA NIM (generation + embeddings)
```

See [docs/architecture.md](docs/architecture.md) for the full Mermaid diagram.

## Quickstart

```bash
cp .env.template .env
# Add NVIDIA_API_KEY to .env

pip install -r requirements.txt

# Start Postgres with pgvector
cd deploy && docker-compose up postgres -d

# Run the API
uvicorn api.server:app --reload --port 8081
```

## Docker

```bash
cd deploy
docker-compose up --build
```

## Ingestion

```python
from ingestion.pdf_loader import PDFLoader
from pipeline.embedder import Embedder
from pipeline.indexer import Indexer

loader = PDFLoader()
docs = loader.load_directory("./data/pdfs")

embedder = Embedder()
indexer = Indexer(backend="pgvector")
indexer.upsert(docs, embedder.as_langchain())
```

## Retrieval & Generation

```python
from pipeline.retriever import HybridRetriever
from pipeline.generator import RAGGenerator

retriever = HybridRetriever(dense_retriever=indexer.get_retriever(embedder.as_langchain()), corpus_docs=docs)
generator = RAGGenerator(retriever)
result = generator.generate_with_sources("What is the refund policy?")
print(result["answer"])
print(result["sources"])
```

## Evaluation

```bash
python -m evals.ragas_eval
python -m evals.langsmith_eval
```

## Key Components

| Module | Description |
|---|---|
| `ingestion/pdf_loader.py` | PyMuPDF + RecursiveCharacterTextSplitter |
| `ingestion/sql_loader.py` | Schema-aware SQL table → Document |
| `ingestion/web_loader.py` | httpx + BeautifulSoup scraper |
| `ingestion/slack_loader.py` | Slack export JSON → Documents |
| `pipeline/embedder.py` | NIM embedding wrapper |
| `pipeline/retriever.py` | BM25 + dense hybrid + cross-encoder rerank |
| `pipeline/generator.py` | NIM-backed RAG generation chain |
| `backends/pgvector_backend.py` | Postgres pgvector store |
| `backends/milvus_backend.py` | Milvus vector store |
| `evals/ragas_eval.py` | RAGAS faithfulness/relevancy/recall eval |
| `evals/langsmith_eval.py` | LangSmith dataset + evaluator harness |

## Environment Variables

| Variable | Description |
|---|---|
| `NVIDIA_API_KEY` | NVIDIA NIM API key |
| `NIM_BASE_URL` | NIM endpoint (default: build.nvidia.com) |
| `VECTOR_BACKEND` | `pgvector` or `milvus` |
| `PGVECTOR_CONNECTION` | SQLAlchemy connection string |
| `LANGSMITH_API_KEY` | LangSmith tracing + evals |

## Cross-Repo Integration

- [`nvidia-nim-agent-toolkit`](https://github.com/TylrDn/nvidia-nim-agent-toolkit) — DocAgent uses this pipeline as its RAG backend
- [`agentic-guardrails-eval`](https://github.com/TylrDn/agentic-guardrails-eval) — safety eval suite can target this pipeline's `/query` endpoint

## Topics

`rag` `retrieval-augmented-generation` `langchain` `pgvector` `milvus` `ragas` `langsmith` `nemo-retriever` `python` `enterprise-ai`
