"""LangGraph RAG orchestration graph."""
from langgraph.graph import StateGraph, END
from orchestrator.state import RAGState
from orchestrator.nodes.query_rewriter import rewrite_query
from orchestrator.nodes.retriever import retrieve_documents
from orchestrator.nodes.grader import grade_documents
from orchestrator.nodes.generator import generate_answer
from orchestrator.nodes.hallucination_checker import check_hallucination


def should_retry(state: RAGState) -> str:
    if state["retry_count"] >= 2:
        return "generate"  # force generation after 2 retries
    if not state["graded_documents"]:
        return "rewrite"   # no relevant docs — rewrite query
    return "generate"


def check_answer(state: RAGState) -> str:
    if state["hallucination_score"] >= 0.6:
        return "end"
    if state["retry_count"] >= 2:
        return "end"       # give up gracefully
    return "retry"


def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("grade", grade_documents)
    graph.add_node("generate", generate_answer)
    graph.add_node("hallucination_check", check_hallucination)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", should_retry, {
        "rewrite": "rewrite_query",
        "generate": "generate",
    })
    graph.add_edge("generate", "hallucination_check")
    graph.add_conditional_edges("hallucination_check", check_answer, {
        "end": END,
        "retry": "rewrite_query",
    })

    return graph.compile()


rag_graph = build_rag_graph()
