"""RAG answer generation.

Two entry points share the same NIM-backed prompting:

* :class:`Generator` — stateless; takes a question plus already-retrieved
  documents and returns a :class:`GenerationResult`. Used by the FastAPI server
  and the CRAG orchestrator.
* :class:`RAGGenerator` — wraps a retriever into an LCEL chain. Convenience
  entry point for notebooks and the eval harness.

Every LLM construction attaches Langfuse callbacks from :mod:`core.observability`.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from core.observability import get_callbacks, get_langfuse_handler

logger = logging.getLogger(__name__)

# ChatNVIDIA is the NIM-native chat model. Imported defensively so the module
# loads (and is patchable in tests) even if the package is unavailable.
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    CHATNVIDIA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the package
    ChatNVIDIA = None  # type: ignore[assignment,misc]
    CHATNVIDIA_AVAILABLE = False

DEFAULT_MODEL = os.getenv("CHAT_MODEL", "meta/llama-3.1-70b-instruct")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

_NO_CONTEXT_REPLY = "I don't have enough context in the provided documents to answer that."

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question using ONLY the "
            "provided context. Cite sources inline as [n]. If the answer is not in the "
            "context, say you don't have enough information.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

_GROUNDING_PROMPT = (
    "You are a factuality checker. Given the context and an answer, respond with a "
    'JSON object {{"grounded": true}} if every claim in the answer is supported by '
    'the context, otherwise {{"grounded": false}}.\n\n'
    "Context:\n{context}\n\nAnswer:\n{answer}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_context(docs: list[Any]) -> str:
    """Render documents as a numbered, source-attributed context block."""
    if not docs:
        return ""
    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        source = doc.metadata.get("source", "unknown") if hasattr(doc, "metadata") else "unknown"
        parts.append(f"[{index}] (source: {source})\n{content}")
    return "\n\n".join(parts)


def _parse_grounded_flag(text: str) -> bool:
    """Parse a grounding judgment; default to ``True`` on malformed input."""
    try:
        return bool(json.loads(text).get("grounded", True))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return True


@dataclass
class GenerationResult:
    """The outcome of a generation call."""

    answer: str
    sources: list[str] = field(default_factory=list)
    grounded: bool = True
    refused: bool = False

    def __str__(self) -> str:
        flags = []
        if self.refused:
            flags.append("REFUSED")
        if not self.grounded:
            flags.append("UNGROUNDED")
        prefix = f"[{','.join(flags)}] " if flags else ""
        return f"{prefix}{self.answer}"


# ---------------------------------------------------------------------------
# Generator — stateless, documents-in / answer-out
# ---------------------------------------------------------------------------


class Generator:
    """Generate grounded answers from a question and retrieved documents."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        hallucination_check: bool = False,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            model: NIM model name.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            hallucination_check: When True, score answer grounding after generation.
            base_url: NIM base URL (falls back to ``NIM_BASE_URL``).
            api_key: NIM API key (falls back to ``NVIDIA_API_KEY``/``NIM_API_KEY``).
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.hallucination_check = hallucination_check
        self.base_url = base_url or NIM_BASE_URL
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY") or ""
        self._llm: Any | None = None

    @property
    def llm(self) -> Any:
        """Lazily construct the NIM chat model with Langfuse callbacks attached."""
        if self._llm is None:
            self._llm = ChatNVIDIA(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                base_url=self.base_url,
                api_key=self.api_key,
                callbacks=get_callbacks(),
            )
        return self._llm

    @staticmethod
    def _extract_sources(docs: list[Any]) -> list[str]:
        seen: list[str] = []
        for doc in docs:
            source = (
                doc.metadata.get("source", "unknown")
                if hasattr(doc, "metadata")
                else "unknown"
            )
            if source not in seen:
                seen.append(source)
        return seen

    @staticmethod
    def _valid_docs(documents: list[Any]) -> list[Any]:
        return [
            d
            for d in documents
            if getattr(d, "page_content", "") and d.page_content.strip()
        ]

    def generate(self, question: str, documents: list[Any]) -> GenerationResult:
        """Answer ``question`` grounded in ``documents``.

        Returns a refused result (no LLM call) when no usable context is present.
        """
        docs = self._valid_docs(documents)
        if not docs:
            return GenerationResult(
                answer=_NO_CONTEXT_REPLY, sources=[], grounded=False, refused=True
            )

        context = _format_context(docs)
        sources = self._extract_sources(docs)
        chain = RAG_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke(
            {"context": context, "question": question},
            config={"callbacks": get_callbacks()},
        )

        grounded = True
        if self.hallucination_check:
            grounded = self._check_grounding(answer, context)

        return GenerationResult(
            answer=answer, sources=sources, grounded=grounded, refused=False
        )

    def stream(self, question: str, documents: list[Any]) -> Iterator[str]:
        """Yield answer tokens for ``question`` grounded in ``documents``."""
        docs = self._valid_docs(documents)
        if not docs:
            yield _NO_CONTEXT_REPLY
            return
        context = _format_context(docs)
        chain = RAG_PROMPT | self.llm | StrOutputParser()
        yield from chain.stream(
            {"context": context, "question": question},
            config={"callbacks": get_callbacks()},
        )

    def _check_grounding(self, answer: str, context: str) -> bool:
        """Return True if the answer is grounded in the context."""
        prompt = _GROUNDING_PROMPT.format(context=context, answer=answer)
        response = self.llm.invoke(prompt, config={"callbacks": get_callbacks()})
        text = response.content if hasattr(response, "content") else str(response)
        return _parse_grounded_flag(text)


# ---------------------------------------------------------------------------
# RAGGenerator — retriever-backed LCEL chain
# ---------------------------------------------------------------------------


class RAGGenerator:
    """Retrieval-augmented generation chain over a LangChain retriever."""

    def __init__(
        self,
        retriever: Any,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the chain.

        Args:
            retriever: Any retriever implementing ``invoke``.
            model: NIM model name.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            base_url: NIM base URL (falls back to ``NIM_BASE_URL``).
            api_key: NIM API key (falls back to ``NVIDIA_API_KEY``).
        """
        self.retriever = retriever
        self.model = model
        self._generator = Generator(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
        )
        handler = get_langfuse_handler()
        base_chain = (
            RunnableParallel(
                context=self.retriever | _format_context,
                question=RunnablePassthrough(),
            )
            | RAG_PROMPT
            | self._generator.llm
            | StrOutputParser()
        )
        self.chain = base_chain.with_config({"callbacks": [handler]}) if handler else base_chain

    def generate(self, query: str) -> str:
        """Answer ``query`` using retrieved context."""
        return self.chain.invoke(query, config={"callbacks": get_callbacks()})

    def generate_with_sources(self, query: str) -> dict[str, Any]:
        """Answer ``query`` and return the answer alongside source documents."""
        source_docs = self.retriever.invoke(query)
        result = self._generator.generate(query, source_docs)
        return {
            "answer": result.answer,
            "sources": source_docs,
            "contexts": [d.page_content for d in source_docs],
            "query": query,
        }
