"""pgvector backend — Postgres-based vector store via LangChain."""
from __future__ import annotations

import os

from langchain_postgres import PGVector

PGVECTOR_CONNECTION = os.getenv(
    "PGVECTOR_CONNECTION",
    "postgresql+psycopg://rag:rag@localhost:5432/ragdb",
)
COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION", "rag_docs")

_store = None


def get_pgvector_store(embeddings) -> PGVector:
    global _store
    if _store is None:
        _store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=PGVECTOR_CONNECTION,
        )
    return _store
