"""Integration-style tests for api/server.py using FastAPI TestClient.

All external I/O (PgVector, NIM, BM25) is mocked so no live services needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document


@pytest.fixture()
def client():
    with patch("api.server.get_embedder") as mock_emb, \
         patch("api.server.get_vector_store") as mock_vs, \
         patch("api.server.Indexer") as mock_idx, \
         patch("api.server.Generator") as mock_gen, \
         patch("api.server.HybridRetriever") as mock_hr_cls, \
         patch("api.server.set_retriever"), \
         patch("api.server.rag_graph") as mock_graph:

        mock_emb.return_value = MagicMock()

        mock_backend = MagicMock()
        mock_backend.health_check.return_value = True
        mock_vs.return_value = mock_backend

        mock_indexer = MagicMock()
        from pipeline.indexer import IndexResult

        mock_indexer.index_async = AsyncMock(
            return_value=IndexResult(total=2, indexed=2, skipped=0, failed=0)
        )
        mock_idx.return_value = mock_indexer

        mock_generator = MagicMock()
        mock_generator.model = "meta/llama3-70b-instruct"
        from pipeline.generator import GenerationResult

        mock_generator.generate.return_value = GenerationResult(
            answer="Test answer.", sources=["test.pdf"], grounded=True, refused=False
        )
        mock_generator.stream.return_value = iter(["token1", " token2"])
        mock_gen.return_value = mock_generator

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_hr_cls.return_value = mock_retriever

        mock_graph.ainvoke = AsyncMock(
            return_value={
                "generation": "CRAG answer.",
                "graded_documents": [
                    Document(page_content="ctx", metadata={"source": "test.pdf"})
                ],
                "web_search_triggered": False,
            }
        )

        from api.server import app

        with TestClient(app) as c:
            yield c


@pytest.fixture(autouse=True)
def mock_nim_health():
    with patch("api.server.check_nim_health", new=AsyncMock(return_value=True)):
        yield


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend_ok"] is True
    assert data["nim_reachable"] is True
    assert data["vectorstore_backend"] == "pgvector"


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


def test_query_success(client):
    resp = client.post("/query", json={"question": "What is NIM?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "CRAG answer."
    assert "sources" in data
    assert "grounded" in data


def test_query_blank_question_returns_400(client):
    resp = client.post("/query", json={"question": "   "})
    assert resp.status_code == 400


def test_stream_returns_sse(client):
    resp = client.get("/query/stream", params={"question": "Tell me about NIM"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
