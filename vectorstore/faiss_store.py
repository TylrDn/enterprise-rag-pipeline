"""FAISS in-memory vector store wrapper (development / offline use only)."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from vectorstore.base import VectorStoreBase

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


class FAISSStore(VectorStoreBase):
    """In-memory FAISS store. Not persisted across restarts — dev use only."""

    def __init__(self, embedder: Embeddings) -> None:
        """Initialize the store.

        Args:
            embedder: Embeddings used to vectorize documents and queries.
        """
        logger.warning(
            "Using FAISS in-memory store — data will not persist across restarts. "
            "Not suitable for production."
        )
        self.embedder = embedder
        self._store: FAISS | None = None

    def upsert_documents(
        self, documents: list[Document], ids: list[str] | None = None
    ) -> list[str]:
        """Insert documents, building the index on first call."""
        if not documents:
            return []
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        if self._store is None:
            self._store = FAISS.from_documents(documents, self.embedder, ids=ids)
        else:
            self._store.add_documents(documents, ids=ids)
        return ids

    def similarity_search(
        self, query: str, k: int = DEFAULT_TOP_K, filter: dict[str, Any] | None = None
    ) -> list[Document]:
        """Return the ``k`` most similar documents to ``query``."""
        if self._store is None:
            return []
        return self._store.similarity_search(query=query, k=k, filter=filter)

    def delete(
        self, ids: list[str] | None = None, filter: dict[str, Any] | None = None
    ) -> None:
        """Delete documents by id."""
        if self._store is not None and ids is not None:
            self._store.delete(ids=ids)

    def as_retriever(self, **kwargs: Any) -> Any:
        """Return a retriever over the store (raises if no documents indexed yet)."""
        if self._store is None:
            raise RuntimeError("FAISS store is empty; index documents before retrieval.")
        search_kwargs = kwargs.pop("search_kwargs", {"k": DEFAULT_TOP_K})
        return self._store.as_retriever(search_kwargs=search_kwargs)

    def health_check(self) -> bool:
        """FAISS is in-process; always reachable."""
        return True

    def save(self, path: str) -> None:
        """Persist the index to ``path`` (best-effort, dev convenience)."""
        if self._store is not None:
            self._store.save_local(path)

    def load(self, path: str) -> "FAISSStore":
        """Load an index previously written by :meth:`save`."""
        self._store = FAISS.load_local(
            path, self.embedder, allow_dangerous_deserialization=True
        )
        return self
