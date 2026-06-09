"""FastAPI gateway for the enterprise RAG pipeline.

Single production entry point. On startup it initializes the embedder, vector
store backend, hybrid retriever, and compiled CRAG graph. Every response carries
an ``X-Request-ID`` and a structured access-log line.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, StreamingResponse

from ingestion import ingest_pdf, ingest_url
from orchestrator.graph import rag_graph
from orchestrator.nodes.retriever import set_retriever
from orchestrator.state import initial_state
from pipeline.embedder import get_embedder
from pipeline.generator import Generator
from pipeline.indexer import Indexer
from pipeline.retriever import HybridRetriever
from vectorstore.factory import get_vector_store

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
access_logger = logging.getLogger("api.access")

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY") or ""
VECTORSTORE_BACKEND = os.getenv("VECTORSTORE_BACKEND") or os.getenv("VECTOR_BACKEND", "pgvector")


async def check_nim_health() -> bool:
    """Return True when the NIM ``/models`` endpoint responds with HTTP 200."""
    if not NIM_API_KEY:
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{NIM_BASE_URL.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {NIM_API_KEY}"},
            )
            return response.status_code == 200
    except Exception:
        logger.debug("NIM health check failed", exc_info=True)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared pipeline components once at startup."""
    embedder = get_embedder()
    backend = get_vector_store(embedder.as_langchain(), backend=VECTORSTORE_BACKEND)
    app.state.embedder = embedder
    app.state.backend = backend
    app.state.indexer = Indexer(backend=backend)
    app.state.generator = Generator()
    app.state.corpus = []
    app.state.retriever = HybridRetriever(
        vector_store=backend,
        corpus_documents=app.state.corpus,
    )
    set_retriever(app.state.retriever)
    app.state.rag_graph = rag_graph
    logger.info("RAG API ready (backend=%s)", VECTORSTORE_BACKEND)
    yield


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Attach a request id, time the handler, and log the outcome."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        access_logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response


app = FastAPI(
    title="Enterprise RAG Pipeline",
    description="Multi-source RAG with hybrid retrieval and CRAG orchestration on NVIDIA NIM.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(StructuredLoggingMiddleware)


class IngestRequest(BaseModel):
    """Request body for ``/ingest``."""

    texts: list[str]
    metadatas: list[dict[str, Any]] | None = None
    dry_run: bool = False


class IngestResponse(BaseModel):
    """Response body for ``/ingest``."""

    total: int
    indexed: int
    skipped: int
    failed: int
    dry_run: bool


class IngestPathRequest(BaseModel):
    """Request body for file/URL ingestion routes."""

    path: str | None = None
    url: str | None = None
    dry_run: bool = False


class QueryRequest(BaseModel):
    """Request body for ``/query``."""

    question: str
    top_k: int | None = None
    use_crag: bool = Field(default=True, description="Use CRAG LangGraph orchestration.")


class QueryResponse(BaseModel):
    """Response body for ``/query``."""

    answer: str
    sources: list[str]
    grounded: bool
    web_search_triggered: bool = False


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness/readiness probe including NIM and vector backend reachability."""
    backend_ok = bool(app.state.backend.health_check())
    nim_reachable = await check_nim_health()
    status = "ok" if backend_ok else "degraded"
    return {
        "status": status,
        "nim_reachable": nim_reachable,
        "vectorstore_backend": VECTORSTORE_BACKEND,
        "backend_ok": backend_ok,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Chunk, deduplicate, and index a batch of raw texts."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="`texts` must not be empty.")
    if request.metadatas is not None and len(request.metadatas) != len(request.texts):
        raise HTTPException(
            status_code=400, detail="`metadatas` length must match `texts` length."
        )

    documents = [
        Document(
            page_content=text,
            metadata=(request.metadatas[i] if request.metadatas else {}),
        )
        for i, text in enumerate(request.texts)
    ]

    result = await app.state.indexer.index_async(documents, dry_run=request.dry_run)
    if not request.dry_run:
        app.state.corpus.extend(documents)
        app.state.retriever.update_corpus(app.state.corpus)

    return IngestResponse(
        total=result.total,
        indexed=result.indexed,
        skipped=result.skipped,
        failed=result.failed,
        dry_run=request.dry_run,
    )


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf_route(request: IngestPathRequest) -> IngestResponse:
    """Ingest a PDF from a local path."""
    if not request.path:
        raise HTTPException(status_code=400, detail="`path` is required.")
    documents = await ingest_pdf(request.path)
    result = await app.state.indexer.index_async(documents, dry_run=request.dry_run)
    if not request.dry_run:
        app.state.corpus.extend(documents)
        app.state.retriever.update_corpus(app.state.corpus)
    return IngestResponse(
        total=result.total,
        indexed=result.indexed,
        skipped=result.skipped,
        failed=result.failed,
        dry_run=request.dry_run,
    )


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url_route(request: IngestPathRequest) -> IngestResponse:
    """Ingest content from a URL."""
    if not request.url:
        raise HTTPException(status_code=400, detail="`url` is required.")
    documents = await ingest_url(request.url)
    result = await app.state.indexer.index_async(documents, dry_run=request.dry_run)
    if not request.dry_run:
        app.state.corpus.extend(documents)
        app.state.retriever.update_corpus(app.state.corpus)
    return IngestResponse(
        total=result.total,
        indexed=result.indexed,
        skipped=result.skipped,
        failed=result.failed,
        dry_run=request.dry_run,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Retrieve relevant context and generate a grounded answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="`question` must not be blank.")

    if request.use_crag:
        try:
            state = initial_state(request.question)
            result = await app.state.rag_graph.ainvoke(state)
        except Exception as exc:
            logger.exception("CRAG query failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        docs = result.get("graded_documents") or result.get("documents", [])
        sources = list({d.metadata.get("source", "unknown") for d in docs})
        answer = result.get("generation", "")
        return QueryResponse(
            answer=answer,
            sources=sources,
            grounded=bool(docs),
            web_search_triggered=bool(result.get("web_search_triggered")),
        )

    retriever = HybridRetriever(
        vector_store=app.state.backend,
        corpus_documents=app.state.corpus,
        top_k=request.top_k or 5,
    )
    try:
        docs = retriever.retrieve(request.question)
        gen_result = app.state.generator.generate(request.question, docs)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=gen_result.answer,
        sources=gen_result.sources,
        grounded=gen_result.grounded,
    )


@app.get("/query/stream")
async def query_stream(question: str) -> StreamingResponse:
    """Stream a grounded answer as Server-Sent Events."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="`question` must not be blank.")

    docs = app.state.retriever.retrieve(question)

    async def event_stream() -> AsyncIterator[str]:
        for token in app.state.generator.stream(question, docs):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
