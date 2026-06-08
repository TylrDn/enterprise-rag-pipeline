"""Unit tests for LangGraph RAG orchestration."""
from orchestrator.graph import should_retry, check_answer
from orchestrator.state import RAGState
from langchain_core.documents import Document


def make_state(**kwargs) -> RAGState:
    base = {
        "question": "test?",
        "rewritten_query": "test?",
        "documents": [],
        "graded_documents": [],
        "generation": "",
        "hallucination_score": 0.0,
        "answer_grade": "",
        "retry_count": 0,
        "source_types": [],
    }
    base.update(kwargs)
    return base


def test_should_retry_no_docs():
    state = make_state(graded_documents=[], retry_count=0)
    assert should_retry(state) == "rewrite"


def test_should_retry_has_docs():
    state = make_state(graded_documents=[Document(page_content="context")], retry_count=0)
    assert should_retry(state) == "generate"


def test_should_retry_max_retries():
    state = make_state(graded_documents=[], retry_count=2)
    assert should_retry(state) == "generate"


def test_check_answer_good_score():
    state = make_state(hallucination_score=0.8, retry_count=0)
    assert check_answer(state) == "end"


def test_check_answer_low_score():
    state = make_state(hallucination_score=0.3, retry_count=0)
    assert check_answer(state) == "retry"


def test_check_answer_max_retries():
    state = make_state(hallucination_score=0.3, retry_count=2)
    assert check_answer(state) == "end"
