"""Hybrid BM25 + dense retriever with optional cross-encoder reranking.

This is the single canonical retriever used by both the FastAPI server and the
CRAG orchestrator. Dense retrieval comes from any :class:`VectorStoreBase`
backend; sparse retrieval is BM25 over an in-memory corpus; results are fused
with an ``EnsembleRetriever`` and optionally reranked by a cross-encoder.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from core.exceptions import RetrievalError

logger = logging.getLogger(__name__)

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() in ("1", "true", "yes")
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "cross-encoder").lower()
NIM_RERANK_MODEL = os.getenv("NIM_RERANK_MODEL", "nvidia/nv-rerankqa-mistral-4b-v3")


class HybridRetriever:
    """BM25 + dense hybrid retrieval with optional cross-encoder reranking."""

    def __init__(
        self,
        vector_store: Any,
        corpus_documents: list[Document],
        top_k: int | None = TOP_K,
        rerank: bool = USE_RERANKER,
        hybrid_alpha: float = DENSE_WEIGHT,
        rerank_top_n: int | None = None,
        reranker_model: str | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            vector_store: A backend exposing ``as_retriever`` (dense retrieval).
            corpus_documents: Documents indexed by BM25 for sparse retrieval.
            top_k: Number of documents to return.
            rerank: Whether to apply a cross-encoder reranker.
            hybrid_alpha: Dense weight in [0, 1]; BM25 weight is ``1 - alpha``.
            rerank_top_n: Documents to keep after reranking (defaults to ``top_k``).
            reranker_model: Cross-encoder model name for reranking.
        """
        self.vector_store = vector_store
        self._corpus_documents = corpus_documents
        self.top_k = top_k if top_k is not None else TOP_K
        self.rerank = rerank
        self.hybrid_alpha = hybrid_alpha
        self.rerank_top_n = rerank_top_n or self.top_k
        self.reranker_model = reranker_model or RERANK_MODEL
        self._retriever: Any | None = None

    def _build(self) -> Any:
        """Construct the underlying ensemble (and reranker, if enabled)."""
        bm25 = BM25Retriever.from_documents(self._corpus_documents, k=self.top_k * 2)
        dense = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        ensemble = EnsembleRetriever(
            retrievers=[bm25, dense],
            weights=[1.0 - self.hybrid_alpha, self.hybrid_alpha],
        )
        if not self.rerank:
            return ensemble

        encoder = HuggingFaceCrossEncoder(model_name=self.reranker_model)
        compressor = CrossEncoderReranker(model=encoder, top_n=self.rerank_top_n)
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble
        )

    def _get_retriever(self) -> Any:
        """Return the cached retriever, building it on first use."""
        if self._retriever is None:
            self._retriever = self._build()
        return self._retriever

    def _nim_rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """Rerank documents using the NIM reranker endpoint."""
        import httpx

        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY") or ""
        base_url = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        passages = [d.page_content for d in docs]
        try:
            response = httpx.post(
                f"{base_url.rstrip('/')}/ranking",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.reranker_model or NIM_RERANK_MODEL,
                    "query": {"text": query},
                    "passages": [{"text": p} for p in passages],
                },
                timeout=30,
            )
            response.raise_for_status()
            rankings = response.json().get("rankings", [])
            ordered = sorted(rankings, key=lambda item: item.get("index", 0))
            return [docs[item["index"]] for item in ordered[: self.rerank_top_n]]
        except Exception as exc:
            logger.warning("NIM reranker failed, returning fusion order: %s", exc)
            return docs[: self.rerank_top_n]

    def retrieve(self, query: str) -> list[Document]:
        """Return documents relevant to ``query``."""
        try:
            docs = self._get_retriever().invoke(query)
        except Exception as exc:
            logger.exception("Hybrid retrieval failed")
            raise RetrievalError(str(exc)) from exc
        if self.rerank and RERANK_BACKEND == "nim" and docs:
            return self._nim_rerank(query, docs)
        return docs

    def retrieve_with_sources(self, query: str) -> list[dict[str, Any]]:
        """Return retrieved documents as ``{content, source, metadata}`` dicts."""
        docs = self.retrieve(query)
        return [
            {
                "content": d.page_content,
                "source": d.metadata.get("source", "unknown"),
                "metadata": d.metadata,
            }
            for d in docs
        ]

    def update_corpus(self, documents: list[Document]) -> None:
        """Replace the BM25 corpus and force a rebuild on next retrieval."""
        self._corpus_documents = documents
        self._retriever = None

    def as_langchain_retriever(self) -> Any:
        """Return the underlying LangChain retriever object."""
        return self._get_retriever()
