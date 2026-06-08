"""FastAPI server for the enterprise RAG pipeline."""
from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Enterprise RAG Pipeline",
    description="Multi-source RAG with pgvector/Milvus backends and RAGAS eval harness",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    include_sources: Optional[bool] = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": os.getenv("VECTOR_BACKEND", "pgvector")}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    # Wire in the RAGGenerator here at startup (via lifespan or dependency injection)
    raise HTTPException(status_code=501, detail="Wire pipeline in lifespan handler")


@app.post("/ingest")
def ingest(source_type: str, path: str) -> dict:
    return {"status": "accepted", "source_type": source_type, "path": path}
