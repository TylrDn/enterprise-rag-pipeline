"""RAG generator — wraps a retriever with an LLM to answer questions.

Langfuse tracing is injected into the ``ChatOpenAI`` constructor and,
when a handler is available, the chain is also wrapped via
``.with_config()`` so every LCEL step is traced.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Optional Langfuse import
# ---------------------------------------------------------------------------
try:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:
    LangfuseCallbackHandler = None  # type: ignore[assignment,misc]
    LANGFUSE_AVAILABLE = False

# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the provided context. If the answer is not in the context, say so.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_langfuse_handler() -> "LangfuseCallbackHandler | None":
    """Return a Langfuse CallbackHandler if credentials are configured.

    Returns ``None`` (and never raises) when:
    - ``langfuse`` package is not installed
    - ``LANGFUSE_PUBLIC_KEY`` env var is not set
    """
    if not LANGFUSE_AVAILABLE:
        return None
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        return LangfuseCallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return None


def _format_context(docs: list[Any]) -> str:
    """Concatenate retrieved document page_content strings."""
    return "\n\n".join(
        doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs
    )


# ---------------------------------------------------------------------------
# RAGGenerator
# ---------------------------------------------------------------------------


class RAGGenerator:
    """Retrieval-augmented generation chain.

    Parameters
    ----------
    retriever:
        Any LangChain-compatible retriever (must implement ``.invoke()``).
    model:
        OpenAI-compatible model name served by the NIM endpoint.
    temperature:
        Sampling temperature forwarded to the LLM.
    max_tokens:
        Maximum tokens in the LLM response.
    base_url:
        NIM / OpenAI-compatible API base URL.  Falls back to
        ``NIM_BASE_URL`` env var, then NVIDIA's hosted endpoint.
    api_key:
        API key.  Falls back to ``NVIDIA_API_KEY`` env var.
    """

    def __init__(
        self,
        retriever: Any,
        model: str = "meta/llama-3.1-8b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or os.getenv(
            "NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")

        # Build LLM — attach Langfuse handler if credentials are present
        handler = _get_langfuse_handler()
        callbacks = [handler] if handler else []

        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_base=self.base_url,
            openai_api_key=self.api_key,
            callbacks=callbacks,
        )

        # Build LCEL chain
        base_chain = (
            RunnableParallel(
                context=self.retriever | _format_context,
                question=RunnablePassthrough(),
            )
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

        # Wrap chain with Langfuse config if handler is available
        if handler:
            self.chain = base_chain.with_config({"callbacks": [handler]})
        else:
            self.chain = base_chain

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, query: str, trace_name: str | None = None) -> str:
        """Answer *query* using retrieved context.

        Parameters
        ----------
        query:
            The user question to answer.
        trace_name:
            Optional Langfuse run/trace name.  When provided and Langfuse is
            active, the trace will carry this label in the Langfuse UI.
        """
        run_config: dict[str, Any] = {}
        handler = _get_langfuse_handler()
        if handler:
            cfg: dict[str, Any] = {"callbacks": [handler]}
            if trace_name:
                cfg["run_name"] = trace_name
            run_config = cfg

        if run_config:
            return self.chain.invoke(query, config=run_config)
        return self.chain.invoke(query)

    def generate_with_sources(
        self, query: str, trace_name: str | None = None
    ) -> dict[str, Any]:
        """Answer *query* and return both the answer and the source documents.

        Parameters
        ----------
        query:
            The user question to answer.
        trace_name:
            Optional Langfuse run/trace name.

        Returns
        -------
        dict with keys:
            ``answer`` (str) — the generated answer.
            ``sources`` (list) — the raw documents returned by the retriever.
            ``query`` (str) — the original query.
        """
        # Retrieve docs separately so we can surface them to the caller
        source_docs = self.retriever.invoke(query)
        context = _format_context(source_docs)

        run_config: dict[str, Any] = {}
        handler = _get_langfuse_handler()
        if handler:
            cfg: dict[str, Any] = {"callbacks": [handler]}
            if trace_name:
                cfg["run_name"] = trace_name
            run_config = cfg

        # Build a context-only chain (retriever already called above)
        answer_chain = RAG_PROMPT | self.llm | StrOutputParser()
        if run_config:
            answer = answer_chain.invoke(
                {"context": context, "question": query}, config=run_config
            )
        else:
            answer = answer_chain.invoke({"context": context, "question": query})

        return {
            "answer": answer,
            "sources": source_docs,
            "query": query,
        }
