"""Corrective RAG (CRAG) graph state.

Canonical shape per the cross-repo LangGraph convention: append-only fields use
``Annotated[list, operator.add]`` so nodes return only the keys they change.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    """State threaded through the CRAG graph."""

    question: str
    documents: Annotated[list[Document], operator.add]
    graded_documents: list[Document]
    generation: str
    web_search_triggered: bool
    grade_scores: Annotated[list[float], operator.add]
    iteration: int
    error: str | None
    source_types: list[str]


def initial_state(question: str) -> RAGState:
    """Return a fresh state for a new question.

    Args:
        question: The user's question.

    Returns:
        RAGState: A zero-initialized state ready for graph invocation.
    """
    return {
        "question": question,
        "documents": [],
        "graded_documents": [],
        "generation": "",
        "web_search_triggered": False,
        "grade_scores": [],
        "iteration": 0,
        "error": None,
        "source_types": [],
    }
