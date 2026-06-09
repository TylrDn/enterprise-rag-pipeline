"""Retrieval node — hybrid dense + BM25 over the configured backend."""
from __future__ import annotations

import logging

from orchestrator.state import RAGState
from pipeline.embedder import get_embedder
from pipeline.retriever import HybridRetriever
from vectorstore.factory import get_vector_store

logger = logging.getLogger(__name__)

_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Return a process-wide hybrid retriever, building it once.

    The BM25 corpus starts empty and is grown via ``update_corpus`` as documents
    are ingested; dense retrieval always reflects the live vector store.
    """
    global _retriever
    if _retriever is None:
        embedder = get_embedder()
        store = get_vector_store(embedder.as_langchain())
        _retriever = HybridRetriever(vector_store=store, corpus_documents=[])
    return _retriever


def set_retriever(retriever: HybridRetriever) -> None:
    """Inject a preconfigured retriever (used by the API lifespan)."""
    global _retriever
    _retriever = retriever


def retrieve_documents(state: RAGState) -> dict:
    """Retrieve candidate documents for the question.

    Returns:
        dict: Partial state with retrieved ``documents`` and their source types.
    """
    query = state["question"]
    try:
        docs = get_retriever().retrieve(query)
    except Exception as exc:
        logger.exception("Retrieval failed")
        return {"documents": [], "source_types": [], "error": str(exc)}
    source_types = list({d.metadata.get("source", "unknown") for d in docs})
    return {"documents": docs, "source_types": source_types}
