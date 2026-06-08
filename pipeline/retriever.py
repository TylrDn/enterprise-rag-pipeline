"""Hybrid BM25 + dense retriever with cross-encoder reranking."""
from __future__ import annotations

import os
from typing import Optional

from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))


class HybridRetriever:
    """BM25 + dense hybrid retrieval with cross-encoder reranking."""

    def __init__(self, dense_retriever, corpus_docs: list[Document]) -> None:
        bm25 = BM25Retriever.from_documents(corpus_docs, k=TOP_K * 2)
        self._ensemble = EnsembleRetriever(
            retrievers=[bm25, dense_retriever],
            weights=[BM25_WEIGHT, DENSE_WEIGHT],
        )
        encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
        compressor = CrossEncoderReranker(model=encoder, top_n=TOP_K)
        self._reranker = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self._ensemble,
        )

    def invoke(self, query: str) -> list[Document]:
        return self._reranker.invoke(query)

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)
