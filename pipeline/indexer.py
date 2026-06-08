"""Upsert documents into pgvector or Milvus vector backends."""
from __future__ import annotations

import os
from typing import Literal

from langchain_core.documents import Document

BACKEND = os.getenv("VECTOR_BACKEND", "pgvector")  # "pgvector" | "milvus"


class Indexer:
    """Upserts document chunks into the configured vector store backend."""

    def __init__(self, backend: Literal["pgvector", "milvus"] = BACKEND) -> None:
        self.backend = backend
        self._store = None

    def _get_store(self, embedder):
        if self._store is not None:
            return self._store
        if self.backend == "pgvector":
            from backends.pgvector_backend import get_pgvector_store
            self._store = get_pgvector_store(embedder)
        else:
            from backends.milvus_backend import get_milvus_store
            self._store = get_milvus_store(embedder)
        return self._store

    def upsert(self, docs: list[Document], embedder) -> int:
        store = self._get_store(embedder)
        store.add_documents(docs)
        return len(docs)

    def get_retriever(self, embedder, search_kwargs: dict | None = None):
        store = self._get_store(embedder)
        return store.as_retriever(search_kwargs=search_kwargs or {"k": 5})
