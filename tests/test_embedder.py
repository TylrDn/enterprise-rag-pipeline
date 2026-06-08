"""Unit tests for pipeline/embedder.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.embedder import (
    HuggingFaceEmbeddings,
    NIMEmbeddings,
    TracedEmbedder,
    get_embedder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_vectors(texts):
    return [[0.1, 0.2, 0.3] for _ in texts]


def _fake_vector(text):
    return [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# TracedEmbedder
# ---------------------------------------------------------------------------

class TestTracedEmbedder:
    def _make(self):
        mock_backend = MagicMock()
        mock_backend.embed_documents.side_effect = _fake_vectors
        mock_backend.embed_query.side_effect = _fake_vector
        return TracedEmbedder(backend=mock_backend, backend_name="test/model")

    def test_embed_documents_returns_vectors(self):
        embedder = self._make()
        result = embedder.embed_documents(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 3

    def test_embed_query_returns_vector(self):
        embedder = self._make()
        result = embedder.embed_query("hello")
        assert len(result) == 3

    def test_langfuse_called_when_keys_set(self):
        mock_backend = MagicMock()
        mock_backend.embed_query.return_value = [0.1, 0.2]
        mock_lf = MagicMock()

        with patch("pipeline.embedder._get_langfuse", return_value=mock_lf):
            embedder = TracedEmbedder(backend=mock_backend, backend_name="test")
            embedder.embed_query("test text")

        mock_lf.generation.assert_called_once()


# ---------------------------------------------------------------------------
# get_embedder factory
# ---------------------------------------------------------------------------

class TestGetEmbedder:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            get_embedder(backend="banana")

    def test_nim_backend_selected(self, monkeypatch):
        monkeypatch.setenv("NIM_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_BACKEND", "nim")

        mock_nim = MagicMock()
        mock_nim.embed_query.return_value = [0.1, 0.2, 0.3]

        with patch("pipeline.embedder.NIMEmbeddings", return_value=mock_nim):
            embedder = get_embedder()

        assert isinstance(embedder, TracedEmbedder)

    def test_huggingface_backend_selected(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BACKEND", "huggingface")

        mock_hf = MagicMock()
        mock_hf.embed_query.return_value = [0.1, 0.2, 0.3]

        with patch("pipeline.embedder.HuggingFaceEmbeddings", return_value=mock_hf):
            embedder = get_embedder()

        assert isinstance(embedder, TracedEmbedder)
