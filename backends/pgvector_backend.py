"""pgvector backend for the enterprise RAG pipeline.

Provides an async-first vector store over PostgreSQL + pgvector.
Supports:
  - Document upsert with metadata
  - Dense vector similarity search (cosine)
  - Metadata-filtered search
  - Document deletion by ID or metadata filter
  - Health check / table initialisation

Configuration is read from environment variables (see .env.template)
or passed directly to PgVectorBackend.__init__.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION = "enterprise_rag"
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# PgVectorBackend
# ---------------------------------------------------------------------------

class PgVectorBackend:
    """Async pgvector-backed document store.

    Args:
        embedder:         An Embeddings instance (from pipeline.embedder).
        connection_str:   SQLAlchemy async connection string.  Defaults to
                          PGVECTOR_URL env var.
        collection_name:  pgvector collection / table name.
        pre_delete_collection: Drop and recreate the collection on init.
                               Useful for test fixtures; never use in prod.
    """

    def __init__(
        self,
        embedder: Embeddings,
        connection_str: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
        pre_delete_collection: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self._connection_str = connection_str or os.environ["PGVECTOR_URL"]
        self._embedder = embedder
        self._pre_delete = pre_delete_collection
        self._store: PGVector | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_store(self) -> PGVector:
        """Lazily initialise the PGVector store (creates tables on first call)."""
        if self._store is None:
            self._store = PGVector(
                embeddings=self._embedder,
                collection_name=self.collection_name,
                connection=self._connection_str,
                distance_strategy=DistanceStrategy.COSINE,
                pre_delete_collection=self._pre_delete,
                use_jsonb=True,
            )
        return self._store

    def health_check(self) -> bool:
        """Verify the database is reachable and the collection table exists.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            store = self._get_store()
            # A similarity search with an empty query and k=1 exercises the
            # full stack: connection, embedding, and vector index.
            store.similarity_search("health check", k=1)
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Embed and upsert documents into the vector store.

        Args:
            documents: List of LangChain Document objects.  Each document
                       should have `page_content` and optionally `metadata`.
            ids:       Optional stable IDs for each document.  Auto-generated
                       UUIDs are used if not provided.

        Returns:
            List of document IDs that were upserted.
        """
        if not documents:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        store = self._get_store()
        store.add_documents(documents=documents, ids=ids)
        return ids

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents by their IDs.

        Args:
            ids: List of document IDs to remove.
        """
        store = self._get_store()
        store.delete(ids=ids)

    def delete_by_filter(self, filter: Dict[str, Any]) -> None:
        """Delete documents matching a metadata filter.

        Args:
            filter: Metadata key/value pairs to match for deletion.
                    Example: {"source": "slack", "channel": "#general"}
        """
        store = self._get_store()
        store.delete(filter=filter)

    # ------------------------------------------------------------------
    # Read / search operations
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Cosine similarity search over embedded documents.

        Args:
            query:  Natural language query string.
            k:      Number of top results to return.
            filter: Optional metadata filter dict (passed to pgvector WHERE clause).

        Returns:
            List of Documents ordered by relevance (most similar first).
        """
        store = self._get_store()
        return store.similarity_search(query=query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Cosine similarity search returning (Document, score) tuples.

        Args:
            query:  Natural language query string.
            k:      Number of top results to return.
            filter: Optional metadata filter dict.

        Returns:
            List of (Document, cosine_distance) tuples; lower score = more similar.
        """
        store = self._get_store()
        return store.similarity_search_with_score(query=query, k=k, filter=filter)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """MMR search for diverse, relevant results (reduces redundancy).

        Args:
            query:       Natural language query string.
            k:           Number of results to return.
            fetch_k:     Candidate pool size before MMR re-ranking.
            lambda_mult: Diversity weight (0 = max diversity, 1 = max relevance).
            filter:      Optional metadata filter dict.

        Returns:
            List of Documents selected by MMR algorithm.
        """
        store = self._get_store()
        return store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Return a LangChain-compatible VectorStoreRetriever.

        Args:
            search_type:   'similarity', 'mmr', or 'similarity_score_threshold'.
            search_kwargs: Keyword args forwarded to the underlying search method
                           (e.g. {"k": 8, "filter": {"source": "pdf"}}).

        Returns:
            A VectorStoreRetriever instance ready for use in LangChain chains.
        """
        store = self._get_store()
        return store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs or {"k": DEFAULT_TOP_K},
        )
