"""Tests for the CRAG graph nodes (grader, generator, retriever)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from orchestrator.nodes import generator as gen_node
from orchestrator.nodes import grader as grade_node
from orchestrator.nodes import retriever as retrieve_node
from orchestrator.state import initial_state


def test_grade_documents_filters_irrelevant():
    docs = [Document(page_content="relevant"), Document(page_content="irrelevant")]
    llm = MagicMock()
    llm.invoke.side_effect = [MagicMock(content="yes"), MagicMock(content="no")]
    with patch.object(grade_node, "get_chat_llm", return_value=llm):
        state = initial_state("q")
        state["documents"] = docs
        out = grade_node.grade_documents(state)
    assert len(out["graded_documents"]) == 1
    assert out["grade_scores"] == [1.0, 0.0]
    assert out["iteration"] == 1


def test_generate_answer_returns_generation():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="The answer.")
    with patch.object(gen_node, "get_chat_llm", return_value=llm):
        state = initial_state("q")
        state["graded_documents"] = [Document(page_content="ctx", metadata={"source": "s"})]
        out = gen_node.generate_answer(state)
    assert out["generation"] == "The answer."


def test_generate_answer_handles_llm_error():
    llm = MagicMock()
    llm.invoke.side_effect = Exception("boom")
    with patch.object(gen_node, "get_chat_llm", return_value=llm):
        state = initial_state("q")
        state["graded_documents"] = [Document(page_content="ctx")]
        out = gen_node.generate_answer(state)
    assert out["generation"] == ""
    assert "error" in out


def test_retrieve_documents_collects_sources():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        Document(page_content="d", metadata={"source": "doc.pdf"})
    ]
    with patch.object(retrieve_node, "get_retriever", return_value=retriever):
        out = retrieve_node.retrieve_documents(initial_state("q"))
    assert len(out["documents"]) == 1
    assert "doc.pdf" in out["source_types"]


def test_retrieve_documents_handles_error():
    with patch.object(retrieve_node, "get_retriever", side_effect=Exception("x")):
        out = retrieve_node.retrieve_documents(initial_state("q"))
    assert out["documents"] == []
    assert "error" in out
