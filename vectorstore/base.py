"""Common vector store abstraction.

All backend adapters (pgvector, Milvus, FAISS) implement :class:`VectorStoreBase`
so the pipeline can swap stores via configuration without code changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document


class VectorStoreBase(ABC):
    """Interface every vector store backend must implement."""

    @abstractmethod
    def upsert_documents(
        self, documents: list[Document], ids: list[str] | None = None
    ) -> list[str]:
        """Insert or update documents and return their ids.

        Args:
            documents: Documents to store.
            ids: Optional explicit ids; generated when omitted.

        Returns:
            list[str]: The ids of the upserted documents.
        """

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[Document]:
        """Return the ``k`` most similar documents to ``query``."""

    @abstractmethod
    def delete(
        self, ids: list[str] | None = None, filter: dict[str, Any] | None = None
    ) -> None:
        """Delete documents by id or by metadata filter."""

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> Any:
        """Return a LangChain retriever backed by this store."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` if the backend is reachable, else ``False``."""

    # -- convenience wrappers shared by all backends -----------------------

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Alias for :meth:`upsert_documents` (LangChain naming compatibility)."""
        return self.upsert_documents(documents)

    def delete_by_ids(self, ids: list[str]) -> None:
        """Delete documents by their ids."""
        self.delete(ids=ids)

    def delete_by_filter(self, filter: dict[str, Any]) -> None:
        """Delete documents matching a metadata filter."""
        self.delete(filter=filter)
