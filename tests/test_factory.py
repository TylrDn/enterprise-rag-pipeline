"""Tests for the vector store factory and backend selection."""
from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings

from vectorstore.base import VectorStoreBase
from vectorstore.factory import _resolve_backend, get_vector_store


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_resolve_prefers_canonical_var(monkeypatch):
    monkeypatch.setenv("VECTORSTORE_BACKEND", "milvus")
    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    assert _resolve_backend(None) == "milvus"


def test_resolve_accepts_legacy_alias(monkeypatch):
    monkeypatch.delenv("VECTORSTORE_BACKEND", raising=False)
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    assert _resolve_backend(None) == "faiss"


def test_resolve_rejects_unknown_backend():
    with pytest.raises(ValueError):
        _resolve_backend("does-not-exist")


@pytest.mark.parametrize("backend", ["pgvector", "milvus", "faiss"])
def test_get_vector_store_returns_base_subclass(backend):
    store = get_vector_store(FakeEmbeddings(), backend=backend)
    assert isinstance(store, VectorStoreBase)
