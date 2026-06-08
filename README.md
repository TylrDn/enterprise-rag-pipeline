# enterprise-rag-pipeline

Production-grade multi-source RAG pipeline with PDF, SQL, web, and Slack ingestion. Built on LangChain, LangGraph, NVIDIA NIM embeddings, pgvector, and evaluated with RAGAS. OpenAI-compatible FastAPI gateway with LangSmith tracing.

## Architecture
```
Ingestion (PDF / SQL / Web / Slack)
        ↓
  Chunking + NIM Embeddings
        ↓
   pgvector / FAISS Store
        ↓
 LangGraph RAG Orchestrator
   (Query → Retrieve → Grade → Generate → Hallucination Check)
        ↓
  FastAPI /query endpoint
        ↓
  RAGAS + LangSmith Eval
```

## Structure
```
enterprise-rag-pipeline/
├── ingestion/           # PDF, SQL, web, Slack loaders
├── embeddings/          # NIM embedding client + chunking
├── vectorstore/         # pgvector + FAISS store wrappers
├── retriever/           # Hybrid retriever (dense + BM25)
├── orchestrator/        # LangGraph RAG graph
│   └── nodes/           # query_rewriter, retriever, grader, generator, hallucination_checker
├── evals/               # RAGAS eval runner + LangSmith datasets
├── api/                 # FastAPI gateway
├── configs/             # YAML configs
├── deploy/              # Docker Compose + K8s
├── tests/               # Unit + integration tests
└── docs/                # Architecture docs
```

## Quick Start
```bash
cp .env.template .env
# Fill in NVIDIA_API_KEY, DATABASE_URL, etc.
docker compose -f deploy/docker-compose.yml up
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is our Q4 revenue?"}'
```

## Eval
```bash
python evals/ragas_runner.py
```
