"""Hybrid retriever combining dense (NIM) + BM25 sparse retrieval."""
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from vectorstore.pgvector_store import PGVectorStore
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class HybridRetriever:
    """Dense + BM25 ensemble retriever."""

    def __init__(self, docs: List[Document] = None, k: int = None, alpha: float = 0.7):
        self.k = k or int(os.getenv("TOP_K", "6"))
        self.alpha = alpha
        self.pg_store = PGVectorStore()
        self.dense_retriever = self.pg_store.as_retriever(k=self.k)

        if docs:
            bm25 = BM25Retriever.from_documents(docs, k=self.k)
            self.retriever = EnsembleRetriever(
                retrievers=[bm25, self.dense_retriever],
                weights=[1 - alpha, alpha],
            )
        else:
            self.retriever = self.dense_retriever

    def invoke(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)
