"""Milvus backend — enterprise-scale vector store."""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus

from vectorstore.base import VectorStoreBase

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_URI = "http://localhost:19530"


class MilvusBackend(VectorStoreBase):
    """Vector store backed by Milvus."""

    def __init__(
        self,
        embedder: Embeddings,
        uri: str | None = None,
        collection: str | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            embedder: Embeddings used to vectorize documents and queries.
            uri: Milvus connection URI. Defaults to ``MILVUS_URI``.
            collection: Collection name. Defaults to ``MILVUS_COLLECTION``.
        """
        self.embedder = embedder
        self.uri: str = uri or os.getenv("MILVUS_URI") or DEFAULT_URI
        self.collection: str = collection or os.getenv("MILVUS_COLLECTION") or "rag_docs"
        self._store: Milvus | None = None

    @property
    def store(self) -> Milvus:
        """Lazily construct and cache the underlying Milvus store."""
        if self._store is None:
            self._store = Milvus(
                embedding_function=self.embedder,
                collection_name=self.collection,
                connection_args={"uri": self.uri},
                auto_id=True,
            )
        return self._store

    def upsert_documents(
        self, documents: list[Document], ids: list[str] | None = None
    ) -> list[str]:
        """Insert documents, returning their ids (generated when not supplied)."""
        if not documents:
            return []
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        self.store.add_documents(documents, ids=ids)
        return ids

    def similarity_search(
        self, query: str, k: int = DEFAULT_TOP_K, filter: dict[str, Any] | None = None
    ) -> list[Document]:
        """Return the ``k`` most similar documents to ``query``."""
        expr = filter if isinstance(filter, str) else None
        return self.store.similarity_search(query=query, k=k, expr=expr)

    def delete(
        self, ids: list[str] | None = None, filter: dict[str, Any] | None = None
    ) -> None:
        """Delete documents by id (Milvus deletes by primary key)."""
        if ids is not None:
            self.store.delete(ids=ids)

    def as_retriever(self, **kwargs: Any) -> Any:
        """Return a similarity retriever over the store."""
        search_kwargs = kwargs.pop("search_kwargs", {"k": DEFAULT_TOP_K})
        return self.store.as_retriever(search_kwargs=search_kwargs)

    def health_check(self) -> bool:
        """Return ``True`` if the store can be constructed, else ``False``."""
        try:
            _ = self.store
            return True
        except Exception:
            logger.exception("milvus health check failed")
            return False


def get_milvus_store(embeddings: Embeddings) -> MilvusBackend:
    """Return a :class:`MilvusBackend` (kept for backward compatibility)."""
    return MilvusBackend(embedder=embeddings)
