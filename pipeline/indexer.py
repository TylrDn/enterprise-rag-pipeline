"""Upsert documents to pgvector or Milvus."""
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from typing import List
import logging

logger = logging.getLogger(__name__)


def index_documents(
    docs: List[Document],
    embedder: Embeddings,
    backend: str = "pgvector",
    collection_name: str = "enterprise_rag",
    **backend_kwargs,
) -> None:
    """Embed and upsert documents into the selected vector backend."""
    if backend == "pgvector":
        from backends.pgvector_backend import PgVectorBackend
        store = PgVectorBackend(embedder=embedder, collection_name=collection_name, **backend_kwargs)
    elif backend == "milvus":
        from backends.milvus_backend import MilvusBackend
        store = MilvusBackend(embedder=embedder, collection_name=collection_name, **backend_kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    store.upsert(docs)
    logger.info(f"indexer: upserted {len(docs)} docs to {backend}/{collection_name}")
