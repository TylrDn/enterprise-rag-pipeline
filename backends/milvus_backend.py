"""Milvus backend — enterprise-scale vector store."""
from __future__ import annotations

import os

from langchain_milvus import Milvus

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_docs")

_store = None


def get_milvus_store(embeddings) -> Milvus:
    global _store
    if _store is None:
        _store = Milvus(
            embedding_function=embeddings,
            collection_name=MILVUS_COLLECTION,
            connection_args={"uri": MILVUS_URI},
        )
    return _store
