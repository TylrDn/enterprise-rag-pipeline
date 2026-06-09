"""Corrective RAG (CRAG) LangGraph orchestration.

Canonical flow::

    retrieve -> grade -> route_generation
                           |- relevant docs ----> generate -> END
                           |- insufficient ------> web_search -> generate -> END

``route_generation`` sends the graph to the web-search fallback only while the
iteration count is below ``MAX_ITERATIONS``, bounding corrective work.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from orchestrator.nodes.generator import generate_answer
from orchestrator.nodes.grader import grade_documents
from orchestrator.nodes.retriever import retrieve_documents
from orchestrator.nodes.web_search import web_search_fallback
from orchestrator.state import RAGState

MAX_ITERATIONS = 3


def route_generation(state: RAGState) -> str:
    """Decide whether to generate directly or fall back to web search.

    Args:
        state: Current graph state.

    Returns:
        str: ``"generate"`` when relevant docs exist or the iteration cap is
        reached, otherwise ``"web_search"``.
    """
    if state.get("graded_documents"):
        return "generate"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "generate"
    return "web_search"


def build_rag_graph():
    """Build and compile the CRAG state graph.

    Returns:
        The compiled LangGraph application.
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("grade", grade_documents)
    graph.add_node("web_search", web_search_fallback)
    graph.add_node("generate", generate_answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_generation,
        {"generate": "generate", "web_search": "web_search"},
    )
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


rag_graph = build_rag_graph()
