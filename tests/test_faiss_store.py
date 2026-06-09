"""Tests for the in-memory FAISS store."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from vectorstore.faiss_store import FAISSStore


class FakeEmbeddings(Embeddings):
    """Deterministic 3-d embeddings for offline FAISS tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.0]


def test_search_before_index_returns_empty():
    store = FAISSStore(FakeEmbeddings())
    assert store.similarity_search("anything") == []


def test_upsert_empty_returns_empty():
    store = FAISSStore(FakeEmbeddings())
    assert store.upsert_documents([]) == []


def test_upsert_then_search():
    store = FAISSStore(FakeEmbeddings())
    ids = store.upsert_documents([Document(page_content="hello world")])
    assert len(ids) == 1
    results = store.similarity_search("hello world", k=1)
    assert len(results) == 1


def test_health_check_is_true():
    assert FAISSStore(FakeEmbeddings()).health_check() is True


def test_as_retriever_raises_when_empty():
    with pytest.raises(RuntimeError):
        FAISSStore(FakeEmbeddings()).as_retriever()
