"""Vector store factory.

Selects the active backend from ``VECTORSTORE_BACKEND`` (canonical) or the
legacy ``VECTOR_BACKEND`` alias, so pipeline code never imports a concrete
backend directly.
"""
from __future__ import annotations

import logging
import os

from langchain_core.embeddings import Embeddings

from vectorstore.base import VectorStoreBase

logger = logging.getLogger(__name__)

VALID_BACKENDS = ("pgvector", "milvus", "faiss")


def _resolve_backend(explicit: str | None) -> str:
    """Resolve the backend name, preferring ``VECTORSTORE_BACKEND``."""
    name = (
        explicit
        or os.getenv("VECTORSTORE_BACKEND")
        or os.getenv("VECTOR_BACKEND")
        or "pgvector"
    ).lower()
    if name not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown vector backend '{name}'. Expected one of {VALID_BACKENDS}."
        )
    return name


def get_vector_store(
    embedder: Embeddings, backend: str | None = None
) -> VectorStoreBase:
    """Return the configured vector store backend.

    Args:
        embedder: Embeddings used by the backend.
        backend: Optional override; otherwise read from the environment.

    Returns:
        VectorStoreBase: An initialized backend adapter.
    """
    name = _resolve_backend(backend)
    logger.info("Initializing vector store backend: %s", name)

    if name == "pgvector":
        from backends.pgvector_backend import PgVectorBackend

        return PgVectorBackend(embedder=embedder)
    if name == "milvus":
        from backends.milvus_backend import MilvusBackend

        return MilvusBackend(embedder=embedder)

    from vectorstore.faiss_store import FAISSStore

    return FAISSStore(embedder=embedder)
