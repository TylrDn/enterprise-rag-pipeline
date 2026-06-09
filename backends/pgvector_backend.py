"""pgvector backend — Postgres-based vector store via LangChain."""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from vectorstore.base import VectorStoreBase

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_CONNECTION = "postgresql+psycopg://rag:rag@localhost:5432/ragdb"


class PgVectorBackend(VectorStoreBase):
    """Vector store backed by Postgres + the pgvector extension."""

    def __init__(
        self,
        embedder: Embeddings,
        connection: str | None = None,
        collection: str | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            embedder: Embeddings used to vectorize documents and queries.
            connection: SQLAlchemy connection string. Defaults to ``PGVECTOR_URL``.
            collection: Collection name. Defaults to ``PGVECTOR_COLLECTION``.
        """
        self.embedder = embedder
        self.connection: str = connection or os.getenv("PGVECTOR_URL") or DEFAULT_CONNECTION
        self.collection: str = collection or os.getenv("PGVECTOR_COLLECTION") or "rag_docs"
        self._store: PGVector | None = None

    @property
    def store(self) -> PGVector:
        """Lazily construct and cache the underlying PGVector store."""
        if self._store is None:
            self._store = PGVector(
                embeddings=self.embedder,
                collection_name=self.collection,
                connection=self.connection,
                use_jsonb=True,
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
        return self.store.similarity_search(query=query, k=k, filter=filter)

    def similarity_search_with_score(
        self, query: str, k: int = DEFAULT_TOP_K, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """Return ``(document, score)`` pairs for the ``k`` nearest documents."""
        return self.store.similarity_search_with_score(query=query, k=k, filter=filter)

    def delete(
        self, ids: list[str] | None = None, filter: dict[str, Any] | None = None
    ) -> None:
        """Delete documents by id or metadata filter."""
        if ids is not None:
            self.store.delete(ids=ids)
        elif filter is not None:
            self.store.delete(filter=filter)

    def as_retriever(self, **kwargs: Any) -> Any:
        """Return a similarity retriever over the store."""
        search_kwargs = kwargs.pop("search_kwargs", {"k": DEFAULT_TOP_K})
        return self.store.as_retriever(
            search_type=kwargs.pop("search_type", "similarity"),
            search_kwargs=search_kwargs,
        )

    def health_check(self) -> bool:
        """Return ``True`` if the store can be constructed, else ``False``."""
        try:
            _ = self.store
            return True
        except Exception:
            logger.exception("pgvector health check failed")
            return False


def get_pgvector_store(embeddings: Embeddings) -> PgVectorBackend:
    """Return a :class:`PgVectorBackend` (kept for backward compatibility)."""
    return PgVectorBackend(embedder=embeddings)
