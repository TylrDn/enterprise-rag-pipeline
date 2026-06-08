"""Integration-style tests for api/server.py using FastAPI TestClient.

All external I/O (PgVector, NIM, BM25) is mocked so no live services needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test")
    monkeypatch.setenv("PGVECTOR_URL", "postgresql+psycopg://u:p@localhost/test")
    monkeypatch.setenv("EMBEDDING_BACKEND", "huggingface")


@pytest.fixture()
def client():
    with patch("api.server.get_embedder") as mock_emb, \
         patch("api.server.PgVectorBackend") as mock_be, \
         patch("api.server.Indexer") as mock_idx, \
         patch("api.server.Generator") as mock_gen:

        mock_emb.return_value = MagicMock()

        mock_backend = MagicMock()
        mock_backend.health_check.return_value = True
        mock_be.return_value = mock_backend

        mock_indexer = MagicMock()
        from pipeline.indexer import IndexResult
        mock_indexer.index.return_value = IndexResult(total=2, indexed=2, skipped=0, failed=0)
        mock_idx.return_value = mock_indexer

        mock_generator = MagicMock()
        mock_generator.model = "meta/llama3-70b-instruct"
        from pipeline.generator import GenerationResult
        mock_generator.generate.return_value = GenerationResult(
            answer="Test answer.", sources=["test.pdf"], grounded=True, refused=False
        )
        mock_generator.stream.return_value = iter(["token1", " token2"])
        mock_gen.return_value = mock_generator

        from api.server import app
        with TestClient(app) as c:
            yield c


# /health
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend_ok"] is True


# /ingest
def test_ingest_success(client):
    resp = client.post("/ingest", json={"texts": ["doc one", "doc two"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["indexed"] == 2


def test_ingest_empty_texts_returns_400(client):
    resp = client.post("/ingest", json={"texts": []})
    assert resp.status_code == 400


def test_ingest_metadata_mismatch_returns_400(client):
    resp = client.post(
        "/ingest",
        json={"texts": ["one"], "metadatas": [{"a": 1}, {"b": 2}]},
    )
    assert resp.status_code == 400


def test_ingest_dry_run(client):
    resp = client.post("/ingest", json={"texts": ["doc"], "dry_run": True})
    assert resp.status_code == 200
    assert resp.json()["dry_run"] is True


# /query
def test_query_success(client):
    with patch("api.server.HybridRetriever") as mock_hr:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            Document(page_content="context", metadata={"source": "test.pdf"})
        ]
        mock_hr.return_value = mock_retriever

        resp = client.post("/query", json={"question": "What is NIM?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert "grounded" in data


def test_query_blank_question_returns_400(client):
    resp = client.post("/query", json={"question": "   "})
    assert resp.status_code == 400


# /query/stream
def test_stream_returns_sse(client):
    with patch("api.server.HybridRetriever") as mock_hr:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_hr.return_value = mock_retriever

        resp = client.get("/query/stream", params={"question": "Tell me about NIM"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
