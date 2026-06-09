"""Unit tests for pipeline/indexer.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from pipeline.indexer import (
    Indexer,
    IndexResult,
    _chunk_documents,
    _content_hash,
    _deduplicate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_backend():
    b = MagicMock()
    b.upsert_documents.return_value = ["id-1"]
    return b


@pytest.fixture()
def indexer(mock_backend, tmp_path):
    """Indexer with a minimal pipeline.yaml written to tmp_path."""
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(
        "chunking:\n  chunk_size: 200\n  chunk_overlap: 20\n"
        "embedding:\n  batch_size: 2\n"
    )
    return Indexer(backend=mock_backend, config_path=cfg)


@pytest.fixture()
def sample_docs():
    return [
        Document(
            page_content="The quick brown fox jumps over the lazy dog. " * 10,
            metadata={"source": "test.pdf"},
        )
    ]


# ---------------------------------------------------------------------------
# _content_hash
# ---------------------------------------------------------------------------

def test_content_hash_is_deterministic():
    assert _content_hash("hello") == _content_hash("hello")


def test_content_hash_differs_for_different_text():
    assert _content_hash("hello") != _content_hash("world")


def test_content_hash_length():
    assert len(_content_hash("anything")) == 16


# ---------------------------------------------------------------------------
# _chunk_documents
# ---------------------------------------------------------------------------

def test_chunk_documents_adds_metadata(sample_docs):
    chunks = _chunk_documents(sample_docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata
        assert "chunk_hash" in chunk.metadata
        assert "source" in chunk.metadata  # original metadata preserved


def test_chunk_documents_respects_chunk_size(sample_docs):
    chunks = _chunk_documents(sample_docs, chunk_size=100, chunk_overlap=10)
    for chunk in chunks:
        assert len(chunk.page_content) <= 120  # allow small splitter overshoot


# ---------------------------------------------------------------------------
# _deduplicate
# ---------------------------------------------------------------------------

def test_deduplicate_no_existing_hashes(sample_docs):
    chunks = _chunk_documents(sample_docs, chunk_size=100, chunk_overlap=10)
    new, seen = _deduplicate(chunks, existing_hashes=None)
    assert len(new) == len(chunks)


def test_deduplicate_removes_known_hashes(sample_docs):
    chunks = _chunk_documents(sample_docs, chunk_size=100, chunk_overlap=10)
    known = {chunks[0].metadata["chunk_hash"]}
    new, _ = _deduplicate(chunks, existing_hashes=known)
    assert len(new) == len(chunks) - 1


# ---------------------------------------------------------------------------
# Indexer.index()
# ---------------------------------------------------------------------------

class TestIndexer:
    def test_index_empty_docs_returns_zero(self, indexer):
        result = indexer.index([])
        assert result.total == 0
        assert result.indexed == 0

    def test_index_returns_index_result(self, indexer, sample_docs):
        result = indexer.index(sample_docs)
        assert isinstance(result, IndexResult)

    def test_index_calls_backend_upsert(self, indexer, mock_backend, sample_docs):
        indexer.index(sample_docs)
        assert mock_backend.upsert_documents.called

    def test_index_dry_run_skips_upsert(self, indexer, mock_backend, sample_docs):
        result = indexer.index(sample_docs, dry_run=True)
        mock_backend.upsert_documents.assert_not_called()
        assert result.dry_run is True
        assert result.indexed == 0

    def test_index_dedup_skips_known_chunks(self, indexer, mock_backend, sample_docs):
        chunks = _chunk_documents(sample_docs, chunk_size=200, chunk_overlap=20)
        all_hashes = {c.metadata["chunk_hash"] for c in chunks}
        result = indexer.index(sample_docs, existing_hashes=all_hashes)
        assert result.skipped == result.total
        mock_backend.upsert_documents.assert_not_called()

    def test_index_counts_failed_batches(self, indexer, mock_backend, sample_docs):
        mock_backend.upsert_documents.side_effect = Exception("DB error")
        result = indexer.index(sample_docs)
        assert result.failed > 0
        assert result.success is False

    def test_index_result_str(self, indexer, sample_docs):
        result = indexer.index(sample_docs)
        assert "IndexResult" in str(result)

    def test_index_result_success_true_when_no_failures(self, indexer, sample_docs):
        result = indexer.index(sample_docs)
        assert result.success is True
