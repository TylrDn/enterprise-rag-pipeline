"""Retrieval node — hybrid dense + BM25."""
from retriever.hybrid_retriever import HybridRetriever
from orchestrator.state import RAGState

_retriever = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def retrieve_documents(state: RAGState) -> RAGState:
    query = state.get("rewritten_query") or state["question"]
    docs = get_retriever().invoke(query)
    source_types = list({d.metadata.get("type", "unknown") for d in docs})
    return {
        **state,
        "documents": docs,
        "source_types": source_types,
    }
