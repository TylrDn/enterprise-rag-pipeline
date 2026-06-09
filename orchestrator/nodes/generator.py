"""Answer generation node using NIM."""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from core.observability import get_callbacks
from orchestrator.nodes._llm import get_chat_llm
from orchestrator.state import RAGState

logger = logging.getLogger(__name__)

GENERATE_PROMPT = """You are a knowledgeable assistant. Answer the question using ONLY the \
provided context.
If the context does not contain enough information, say so clearly.
Cite the source of each fact using [source: <source>] notation.

Context:
{context}

Question: {question}

Answer:"""


def _format_context(docs: list[Document]) -> str:
    """Render documents with inline source attribution."""
    return "\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


def generate_answer(state: RAGState) -> dict:
    """Generate a grounded answer from graded (or web) documents.

    Returns:
        dict: Partial state with the ``generation`` text.
    """
    docs = state.get("graded_documents") or state.get("documents", [])
    question = state["question"]
    context = _format_context(docs)
    prompt = GENERATE_PROMPT.format(context=context, question=question)
    try:
        answer = str(
            get_chat_llm(temperature=0.1)
            .invoke(prompt, config={"callbacks": get_callbacks()})
            .content
        ).strip()
    except Exception as exc:
        logger.exception("Generation failed")
        return {"generation": "", "error": str(exc)}
    return {"generation": answer}
