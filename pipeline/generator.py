"""LLM generation layer for the enterprise RAG pipeline.

Provides a grounded answer generator backed by NVIDIA NIM (OpenAI-compatible
endpoint).  Key features:

  - Structured RAG prompt with injected context + question
  - Streaming and non-streaming response modes
  - Source citation passthrough (returns which docs were used)
  - Grounding guard: refuses to answer when no relevant context is found
  - Hallucination threshold check via a lightweight self-critique call
  - Configurable via configs/rag.yaml (chat section)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "rag.yaml"


def _load_chat_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return (yaml.safe_load(f) or {}).get("chat", {})
    return {}


def _load_grader_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return (yaml.safe_load(f) or {}).get("grader", {})
    return {}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_RAG_SYSTEM = """You are a precise, factual assistant.  Answer the user's
question using ONLY the provided context passages.  If the context does not
contain enough information to answer confidently, respond with exactly:
"I don't have enough information to answer that."

Do not fabricate facts, invent citations, or answer from prior knowledge."""

_RAG_HUMAN = """Context passages:
{context}

---
Question: {question}"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _RAG_SYSTEM), ("human", _RAG_HUMAN)]
)

_HALLUCINATION_SYSTEM = """You are a strict factual grader.
Given a context and an answer, respond with a single JSON object:
{{"grounded": true}} if every claim in the answer is supported by the context,
{{"grounded": false}} otherwise.  No explanation."""

_HALLUCINATION_HUMAN = """Context:
{context}

Answer:
{answer}"""

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _HALLUCINATION_SYSTEM), ("human", _HALLUCINATION_HUMAN)]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(documents: List[Document]) -> str:
    """Concatenate document chunks into a numbered context string."""
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def _parse_grounded_flag(response_text: str) -> bool:
    """Safely parse {\"grounded\": true/false} from LLM response."""
    import json
    import re
    match = re.search(r'\{.*?\}', response_text, re.DOTALL)
    if match:
        try:
            return bool(json.loads(match.group()).get("grounded", True))
        except json.JSONDecodeError:
            pass
    # Default to grounded=True to avoid false refusals on parse failure
    return True


_NO_CONTEXT_REPLY = "I don't have enough information to answer that."


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Structured output from a generation call."""

    answer: str
    sources: List[str] = field(default_factory=list)
    grounded: bool = True
    refused: bool = False  # True when no context was available

    def __str__(self) -> str:
        tag = " [REFUSED]" if self.refused else (" [UNGROUNDED]" if not self.grounded else "")
        return f"GenerationResult{tag}: {self.answer[:120]}"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    """Grounded answer generator backed by NVIDIA NIM.

    Args:
        model:           NIM model name (default: from rag.yaml chat.model).
        base_url:        NIM API base URL.
        temperature:     Sampling temperature (0.0 = deterministic).
        max_tokens:      Max tokens in the generated answer.
        api_key:         NVIDIA API key.  Defaults to NVIDIA_API_KEY env var.
        hallucination_check: Run a secondary grounding-check call after generation.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        hallucination_check: bool = True,
    ) -> None:
        cfg = _load_chat_config()
        grader_cfg = _load_grader_config()

        self.model: str = model or cfg.get("model", "meta/llama3-70b-instruct")
        self.base_url: str = base_url or cfg.get(
            "base_url", "https://integrate.api.nvidia.com/v1"
        )
        self.temperature: float = (
            temperature if temperature is not None else float(cfg.get("temperature", 0.1))
        )
        self.max_tokens: int = (
            max_tokens if max_tokens is not None else int(cfg.get("max_tokens", 1024))
        )
        self.hallucination_check = hallucination_check
        self._hallucination_threshold: float = float(
            grader_cfg.get("hallucination_threshold", 0.6)
        )
        self._api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self._llm: Optional[ChatNVIDIA] = None

    # ------------------------------------------------------------------
    # LLM factory
    # ------------------------------------------------------------------

    def _get_llm(self, streaming: bool = False) -> ChatNVIDIA:
        """Return a (possibly cached) ChatNVIDIA instance."""
        if self._llm is None or self._llm.streaming != streaming:
            self._llm = ChatNVIDIA(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                nvidia_api_key=self._api_key,
                streaming=streaming,
            )
        return self._llm

    # ------------------------------------------------------------------
    # Context guard
    # ------------------------------------------------------------------

    def _has_context(self, documents: List[Document]) -> bool:
        return bool(documents) and any(
            doc.page_content.strip() for doc in documents
        )

    # ------------------------------------------------------------------
    # Hallucination check
    # ------------------------------------------------------------------

    def _check_grounding(self, context: str, answer: str) -> bool:
        """Secondary LLM call to verify the answer is grounded in context."""
        try:
            llm = self._get_llm(streaming=False)
            chain = HALLUCINATION_PROMPT | llm | StrOutputParser()
            response = chain.invoke({"context": context, "answer": answer})
            return _parse_grounded_flag(response)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hallucination check failed (%s); defaulting to grounded=True.", exc)
            return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        documents: List[Document],
    ) -> GenerationResult:
        """Generate a grounded answer from retrieved documents.

        Args:
            question:  The user's natural language question.
            documents: Retrieved context documents (from HybridRetriever).

        Returns:
            GenerationResult with answer text, sources, and grounding flags.
        """
        sources = list({
            doc.metadata.get("source", "unknown") for doc in documents
        })

        if not self._has_context(documents):
            logger.warning("No context documents provided — refusing to answer.")
            return GenerationResult(
                answer=_NO_CONTEXT_REPLY,
                sources=[],
                grounded=True,
                refused=True,
            )

        context = _format_context(documents)
        llm = self._get_llm(streaming=False)
        chain = RAG_PROMPT | llm | StrOutputParser()

        answer = chain.invoke({"context": context, "question": question})
        logger.info("Generated answer (%d chars) from %d docs.", len(answer), len(documents))

        grounded = True
        if self.hallucination_check:
            grounded = self._check_grounding(context, answer)
            if not grounded:
                logger.warning("Hallucination check failed for question: '%s'", question[:80])

        return GenerationResult(
            answer=answer,
            sources=sources,
            grounded=grounded,
            refused=False,
        )

    def stream(
        self,
        question: str,
        documents: List[Document],
    ) -> Iterator[str]:
        """Stream answer tokens as they are generated.

        Note: Hallucination check is skipped in streaming mode.

        Args:
            question:  The user's natural language question.
            documents: Retrieved context documents.

        Yields:
            Answer token strings as they arrive from the LLM.
        """
        if not self._has_context(documents):
            yield _NO_CONTEXT_REPLY
            return

        context = _format_context(documents)
        llm = self._get_llm(streaming=True)
        chain = RAG_PROMPT | llm | StrOutputParser()

        for chunk in chain.stream({"context": context, "question": question}):
            yield chunk

    def build_lcel_chain(self, retriever):
        """Return a simple LCEL chain: retriever | prompt | llm | parser.

        Suitable for use in LangServe or notebook demos where you don't need
        the full GenerationResult envelope.

        Args:
            retriever: A LangChain BaseRetriever (e.g. from HybridRetriever.as_langchain_retriever()).

        Returns:
            A Runnable that accepts a question string and returns an answer string.
        """
        llm = self._get_llm(streaming=False)
        return (
            {"context": retriever | _format_context, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
