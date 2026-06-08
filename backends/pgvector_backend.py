"""pgvector backend — LangChain PGVector wrapper."""
from langchain_postgres import PGVector
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from typing import List
import os


class PgVectorBackend:
    """Thin wrapper around LangChain PGVector for upsert and retrieval."""

    def __init__(self, embedder: Embeddings, collection_name: str = "enterprise_rag", **kwargs):
        connection_string = kwargs.get(
            "connection_string",
            os.environ.get(
                "PGVECTOR_URL",
                "postgresql+psycopg://rag_user:rag_pass@localhost:5432/rag_db"
            )
        )
        self._store = PGVector(
            embeddings=embedder,
            collection_name=collection_name,
            connection=connection_string,
        )

    def upsert(self, docs: List[Document]) -> None:
        self._store.add_documents(docs)

    def as_retriever(self, k: int = 10):
        return self._store.as_retriever(search_kwargs={"k": k})
