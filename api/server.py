"""FastAPI server for the enterprise RAG pipeline.

Endpoints:
  POST /ingest          — load and index documents into pgvector
  POST /query           — hybrid retrieve + generate a grounded answer
  GET  /query/stream    — streaming SSE version of /query
  GET  /health          — liveness / readiness probe

All heavy objects (embedder, backend, indexer, retriever, generator) are
built once at startup and injected via FastAPI dependency injection.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backends.pgvector_backend import PgVectorBackend
from pipeline.embedder import get_embedder
from pipeline.generator import GenerationResult, Generator
from pipeline.indexer import IndexResult, Indexer
from pipeline.retriever import HybridRetriever

load_dotenv()
logger = logging.getLogger(__name__)


class _AppState:
    embedder = None
    backend: Optional[PgVectorBackend] = None
    indexer: Optional[Indexer] = None
    generator: Optional[Generator] = None
    corpus_documents: list = []


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build all singletons once on startup."""
    logger.info("Initialising RAG pipeline components...")
    _state.embedder = get_embedder()
    _state.backend = PgVectorBackend(embedder=_state.embedder)
    _state.indexer = Indexer(backend=_state.backend)
    _state.generator = Generator()
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Enterprise RAG Pipeline",
    description="Hybrid BM25 + dense retrieval with NVIDIA NIM generation.",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_retriever() -> HybridRetriever:
    return HybridRetriever(
        vector_store=_state.backend,
        corpus_documents=_state.corpus_documents,
    )


class IngestRequest(BaseModel):
    texts: List[str] = Field(..., description="Raw text passages to index.")
    metadatas: Optional[List[dict]] = Field(None, description="Per-passage metadata dicts (same length as texts).")
    dry_run: bool = Field(False, description="Chunk and dedup without writing to the store.")


class IngestResponse(BaseModel):
    total: int
    indexed: int
    skipped: int
    failed: int
    dry_run: bool


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question.")
    top_k: Optional[int] = Field(None, description="Override retriever top_k.")
    stream: bool = Field(False, description="Use streaming endpoint instead.")


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    grounded: bool
    refused: bool


class HealthResponse(BaseModel):
    status: str
    backend_ok: bool
    model: str


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest(req: IngestRequest) -> IngestResponse:
    """Chunk, deduplicate, embed, and store text passages."""
    from langchain_core.documents import Document

    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list must not be empty.")

    metadatas = req.metadatas or [{} for _ in req.texts]
    if len(metadatas) != len(req.texts):
        raise HTTPException(status_code=400, detail="metadatas length must match texts length.")

    documents = [Document(page_content=t, metadata=m) for t, m in zip(req.texts, metadatas)]
    result: IndexResult = _state.indexer.index(documents, dry_run=req.dry_run)

    if not req.dry_run:
        _state.corpus_documents.extend(documents)

    return IngestResponse(
        total=result.total,
        indexed=result.indexed,
        skipped=result.skipped,
        failed=result.failed,
        dry_run=result.dry_run,
    )


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query(
    req: QueryRequest,
    retriever: HybridRetriever = Depends(_get_retriever),
) -> QueryResponse:
    """Retrieve context and generate a grounded answer."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be blank.")

    if req.top_k is not None:
        retriever.top_k = req.top_k

    docs = retriever.retrieve(req.question)
    result: GenerationResult = _state.generator.generate(question=req.question, documents=docs)

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        grounded=result.grounded,
        refused=result.refused,
    )


@app.get("/query/stream")
def query_stream(
    question: str,
    retriever: HybridRetriever = Depends(_get_retriever),
) -> StreamingResponse:
    """Server-Sent Events streaming endpoint."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must not be blank.")

    docs = retriever.retrieve(question)

    def token_generator():
        for token in _state.generator.stream(question=question, documents=docs):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + readiness probe."""
    backend_ok = _state.backend.health_check() if _state.backend else False
    return HealthResponse(
        status="ok" if backend_ok else "degraded",
        backend_ok=backend_ok,
        model=_state.generator.model if _state.generator else "uninitialized",
    )
