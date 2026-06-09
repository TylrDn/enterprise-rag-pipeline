"""Unit tests for backends/pgvector_backend.py.

All PGVector and DB calls are mocked — no live database required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from backends.pgvector_backend import DEFAULT_TOP_K, PgVectorBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_embedder():
    emb = MagicMock()
    emb.embed_query.return_value = [0.1, 0.2, 0.3]
    emb.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    return emb


@pytest.fixture()
def backend(mock_embedder, monkeypatch):
    monkeypatch.setenv("PGVECTOR_URL", "postgresql+psycopg://user:pw@localhost/test")
    return PgVectorBackend(embedder=mock_embedder)


@pytest.fixture()
def mock_store():
    store = MagicMock()
    store.similarity_search.return_value = [
        Document(page_content="result doc", metadata={"source": "test"})
    ]
    store.similarity_search_with_score.return_value = [
        (Document(page_content="result doc", metadata={"source": "test"}), 0.12)
    ]
    store.add_documents.return_value = None
    store.delete.return_value = None
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPgVectorBackend:
    def test_upsert_returns_ids(self, backend, mock_store):
        with patch("backends.pgvector_backend.PGVector", return_value=mock_store):
            docs = [Document(page_content="hello", metadata={"source": "pdf"})]
            ids = backend.upsert_documents(docs)
        assert len(ids) == 1
        mock_store.add_documents.assert_called_once()

    def test_upsert_empty_list_returns_empty(self, backend):
        ids = backend.upsert_documents([])
        assert ids == []

    def test_upsert_respects_provided_ids(self, backend, mock_store):
        with patch("backends.pgvector_backend.PGVector", return_value=mock_store):
            docs = [Document(page_content="hello")]
            ids = backend.upsert_documents(docs, ids=["custom-id-123"])
        assert ids == ["custom-id-123"]

    def test_similarity_search_returns_docs(self, backend, mock_store):
        with patch("backends.pgvector_backend.PGVector", return_value=mock_store):
            results = backend.similarity_search("test query", k=3)
        assert len(results) == 1
        assert results[0].page_content == "result doc"
        mock_store.similarity_search.assert_called_once_with(
            query="test query", k=3, filter=None
        )

    def test_similarity_search_with_score(self, backend, mock_store):
        with patch("backends.pgvector_backend.PGVector", return_value=mock_store):
            results = backend.similarity_search_with_score("test query")
        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(score, float)

    def test_delete_by_ids(self, backend, mock_store):
        with patch("backends.pgvector_backend.PGVector", return_value=mock_store):
            backend.delete_by_ids(["id-1", "id-2"])
        mock_store.delete.assert_called_once_with(ids=["id-1", "id-2"])

    def test_delete_by_filter(self, backend, mock_store):
        with patch("backends.pgvector_backend.PGVector", return_value=mock_store):
            backend.delete_by_filter({"source": "slack"})
        mock_store.delete.assert_called_once_with(filter={"source": "slack"})

    def test_health_check_true_on_success(self, backend, mock_store):
        with patch("backends.pgvector_backend.PGVector", return_value=mock_store):
            assert backend.health_check() is True

    def test_health_check_false_on_exception(self, backend):
        with patch(
            "backends.pgvector_backend.PGVector",
            side_effect=Exception("DB down"),
        ):
            assert backend.health_check() is False

    def test_as_retriever_returns_retriever(self, backend, mock_store):
        mock_store.as_retriever.return_value = MagicMock()
        with patch("backends.pgvector_backend.PGVector", return_value=mock_store):
            retriever = backend.as_retriever()
        mock_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": DEFAULT_TOP_K},
        )
        assert retriever is not None
