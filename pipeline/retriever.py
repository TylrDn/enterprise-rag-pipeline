"""Hybrid retriever for the enterprise RAG pipeline.

Combines sparse BM25 retrieval with dense vector search using a weighted
Reciprocal Rank Fusion (RRF) merge, then optionally reranks the fused
candidate set with a cross-encoder model.

Architecture:

  query
    ├─► BM25Retriever          (sparse, lexical)
    ├─► VectorStore.as_retriever (dense, semantic)
    │
    ├─► EnsembleRetriever      (RRF fusion, weighted by hybrid_alpha)
    │
    └─► CrossEncoderReranker   (optional, model from config)
           └─► top-n docs returned to caller

Configuration is read from configs/rag.yaml (retriever section).
All parameters can also be passed directly to HybridRetriever.__init__.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "rag.yaml"


def _load_retriever_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
            return data.get("retriever", {})
    return {}


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Weighted BM25 + dense retriever with optional cross-encoder reranking.

    Args:
        vector_store:     A PgVectorBackend (or any backend exposing .as_retriever()).
        corpus_documents: Documents to build the in-memory BM25 index from.
                          Typically the same docs already indexed in the vector store.
        top_k:            Number of candidates retrieved per leg before fusion.
        rerank:           Whether to apply cross-encoder reranking after fusion.
        rerank_top_n:     Final number of documents returned after reranking.
        hybrid_alpha:     Weight for dense leg in EnsembleRetriever (0=BM25, 1=dense).
                          BM25 weight is automatically 1 - hybrid_alpha.
        reranker_model:   HuggingFace cross-encoder model name or path.
    """

    def __init__(
        self,
        vector_store,
        corpus_documents: List[Document],
        top_k: Optional[int] = None,
        rerank: Optional[bool] = None,
        rerank_top_n: int = 4,
        hybrid_alpha: Optional[float] = None,
        reranker_model: Optional[str] = None,
    ) -> None:
        cfg = _load_retriever_config()

        self.top_k: int = top_k if top_k is not None else int(cfg.get("top_k", 6))
        self.rerank: bool = rerank if rerank is not None else bool(cfg.get("rerank", True))
        self.rerank_top_n: int = rerank_top_n
        self.hybrid_alpha: float = (
            hybrid_alpha if hybrid_alpha is not None
            else float(cfg.get("hybrid_alpha", 0.7))
        )
        self.reranker_model: str = (
            reranker_model
            or os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )

        self._vector_store = vector_store
        self._corpus_documents = corpus_documents
        self._retriever: Optional[BaseRetriever] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> BaseRetriever:
        """Lazily construct the full retriever chain."""
        # --- Sparse leg ---
        bm25 = BM25Retriever.from_documents(
            self._corpus_documents,
            k=self.top_k,
        )
        bm25.k = self.top_k

        # --- Dense leg ---
        dense = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

        # --- Fusion (RRF) ---
        sparse_weight = round(1.0 - self.hybrid_alpha, 4)
        dense_weight = round(self.hybrid_alpha, 4)
        ensemble = EnsembleRetriever(
            retrievers=[bm25, dense],
            weights=[sparse_weight, dense_weight],
        )

        if not self.rerank:
            logger.debug(
                "Reranking disabled. Returning EnsembleRetriever "
                "(alpha=%.2f, top_k=%d).",
                self.hybrid_alpha, self.top_k,
            )
            return ensemble

        # --- Cross-encoder reranker ---
        cross_encoder = HuggingFaceCrossEncoder(model_name=self.reranker_model)
        compressor = CrossEncoderReranker(
            model=cross_encoder,
            top_n=self.rerank_top_n,
        )
        reranking_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble,
        )
        logger.debug(
            "Reranker enabled: model=%s, top_n=%d.",
            self.reranker_model, self.rerank_top_n,
        )
        return reranking_retriever

    def _get_retriever(self) -> BaseRetriever:
        if self._retriever is None:
            self._retriever = self._build()
        return self._retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Document]:
        """Run the full hybrid + rerank pipeline for a query.

        Args:
            query: Natural language query string.

        Returns:
            Ordered list of Documents (most relevant first).
        """
        retriever = self._get_retriever()
        docs = retriever.invoke(query)
        logger.info(
            "Retrieved %d docs for query: '%s'",
            len(docs), query[:80],
        )
        return docs

    def retrieve_with_sources(self, query: str) -> List[dict]:
        """Retrieve documents and return as structured dicts with source metadata.

        Returns:
            List of dicts, each with 'content', 'source', and 'metadata' keys.
        """
        docs = self.retrieve(query)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "metadata": doc.metadata,
            }
            for doc in docs
        ]

    def update_corpus(self, documents: List[Document]) -> None:
        """Replace the BM25 corpus and force rebuild on next retrieve() call.

        Call this after indexing new documents to keep the sparse index in sync.

        Args:
            documents: Full updated document corpus.
        """
        self._corpus_documents = documents
        self._retriever = None  # trigger lazy rebuild
        logger.info("BM25 corpus updated (%d docs). Retriever will rebuild.", len(documents))

    def as_langchain_retriever(self) -> BaseRetriever:
        """Return the underlying LangChain BaseRetriever for use in LCEL chains.

        Example usage in a chain::

            from langchain_core.runnables import RunnablePassthrough
            chain = (
                {"context": retriever.as_langchain_retriever(), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        """
        return self._get_retriever()
