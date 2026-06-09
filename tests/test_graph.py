"""Unit tests for the CRAG LangGraph orchestration routers."""
from __future__ import annotations

from langchain_core.documents import Document

from orchestrator.graph import MAX_ITERATIONS, build_rag_graph, route_generation
from orchestrator.state import RAGState, initial_state


def make_state(**kwargs) -> RAGState:
    state = initial_state("test?")
    state.update(kwargs)
    return state


def test_route_generation_with_relevant_docs():
    state = make_state(graded_documents=[Document(page_content="context")], iteration=0)
    assert route_generation(state) == "generate"


def test_route_generation_no_docs_routes_to_web_search():
    state = make_state(graded_documents=[], iteration=0)
    assert route_generation(state) == "web_search"


def test_route_generation_respects_iteration_cap():
    state = make_state(graded_documents=[], iteration=MAX_ITERATIONS)
    assert route_generation(state) == "generate"


def test_initial_state_defaults():
    state = initial_state("hello?")
    assert state["question"] == "hello?"
    assert state["iteration"] == 0
    assert state["web_search_triggered"] is False
    assert state["documents"] == []


def test_build_rag_graph_compiles():
    graph = build_rag_graph()
    assert graph is not None
