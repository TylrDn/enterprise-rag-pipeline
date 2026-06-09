"""Document relevance grader node."""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from core.observability import get_callbacks
from orchestrator.nodes._llm import get_chat_llm
from orchestrator.state import RAGState

logger = logging.getLogger(__name__)

GRADE_PROMPT = """Is the following document relevant to answering the question?
Answer only 'yes' or 'no'.

Question: {question}
Document: {document}"""


def _grade_one(question: str, doc: Document) -> bool:
    """Return True if ``doc`` is relevant to ``question``."""
    prompt = GRADE_PROMPT.format(question=question, document=doc.page_content[:800])
    response = get_chat_llm(temperature=0.0).invoke(
        prompt, config={"callbacks": get_callbacks()}
    )
    result = str(response.content).strip().lower()
    return result.startswith("yes")


def grade_documents(state: RAGState) -> dict:
    """Grade retrieved documents for relevance.

    Returns:
        dict: Partial state with ``graded_documents``, ``grade_scores``, and an
        incremented ``iteration``.
    """
    question = state["question"]
    docs = state.get("documents", [])
    graded: list[Document] = []
    scores: list[float] = []
    for doc in docs:
        relevant = _grade_one(question, doc)
        scores.append(1.0 if relevant else 0.0)
        if relevant:
            graded.append(doc)
    return {
        "graded_documents": graded,
        "grade_scores": scores,
        "iteration": state.get("iteration", 0) + 1,
    }
