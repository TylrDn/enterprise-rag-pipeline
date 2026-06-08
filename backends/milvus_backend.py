"""Milvus backend — LangChain Milvus wrapper."""
from langchain_milvus import Milvus
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from typing import List
import os


class MilvusBackend:
    """Thin wrapper around LangChain Milvus for upsert and retrieval."""

    def __init__(self, embedder: Embeddings, collection_name: str = "enterprise_rag", **kwargs):
        uri = kwargs.get("uri", os.environ.get("MILVUS_URI", "http://localhost:19530"))
        self._store = Milvus(
            embedding_function=embedder,
            collection_name=collection_name,
            connection_args={"uri": uri},
            auto_id=True,
        )

    def upsert(self, docs: List[Document]) -> None:
        self._store.add_documents(docs)

    def as_retriever(self, k: int = 10):
        return self._store.as_retriever(search_kwargs={"k": k})
