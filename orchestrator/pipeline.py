"""End-to-end RAG pipeline orchestrator.

RAGPipeline wires all components together for use outside the HTTP server
(e.g. notebooks, scripts, eval harnesses).
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from pipeline.embedder import get_embedder
from pipeline.generator import GenerationResult, Generator
from pipeline.indexer import Indexer, IndexResult
from pipeline.retriever import HybridRetriever
from vectorstore.base import VectorStoreBase
from vectorstore.factory import get_vector_store


class RAGPipeline:
    """Convenience wrapper that composes Indexer + HybridRetriever + Generator.

    Args:
        embedder:    Embeddings instance (default: from get_embedder()).
        backend:     Vector backend (default: PgVectorBackend).
        indexer:     Indexer instance.
        generator:   Generator instance.
    """

    def __init__(
        self,
        embedder=None,
        backend: Optional[VectorStoreBase] = None,
        indexer: Optional[Indexer] = None,
        generator: Optional[Generator] = None,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.backend = backend or get_vector_store(self.embedder.as_langchain())
        self.indexer = indexer or Indexer(backend=self.backend)
        self.generator = generator or Generator()
        self._corpus: List[Document] = []

    def index(self, documents: List[Document], **kwargs) -> IndexResult:
        """Index documents and update the BM25 corpus."""
        result = self.indexer.index(documents, **kwargs)
        if not kwargs.get("dry_run"):
            self._corpus.extend(documents)
        return result

    def query(self, question: str, top_k: Optional[int] = None) -> GenerationResult:
        """Retrieve + generate a grounded answer."""
        retriever = HybridRetriever(
            vector_store=self.backend,
            corpus_documents=self._corpus,
            top_k=top_k,
        )
        docs = retriever.retrieve(question)
        return self.generator.generate(question=question, documents=docs)

    def answer_with_contexts(
        self, question: str, top_k: Optional[int] = None
    ) -> dict:
        """Retrieve + generate, returning answer and the retrieved context texts.

        Used by the eval harnesses (RAGAS needs the raw contexts, not just the
        source names that :class:`GenerationResult` exposes).

        Returns:
            dict: ``{"answer": str, "contexts": list[str], "sources": list[str]}``.
        """
        retriever = HybridRetriever(
            vector_store=self.backend,
            corpus_documents=self._corpus,
            top_k=top_k,
        )
        docs = retriever.retrieve(question)
        result = self.generator.generate(question=question, documents=docs)
        return {
            "answer": result.answer,
            "contexts": [d.page_content for d in docs],
            "sources": result.sources,
        }

    def stream(self, question: str):
        """Stream answer tokens for a question."""
        retriever = HybridRetriever(
            vector_store=self.backend,
            corpus_documents=self._corpus,
        )
        docs = retriever.retrieve(question)
        yield from self.generator.stream(question=question, documents=docs)
