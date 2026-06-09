"""Web search fallback node for Corrective RAG.

When document grading finds no relevant context, the graph routes here to fetch
fresh context from the web before generation. Uses Tavily when ``TAVILY_API_KEY``
is set, otherwise a keyless DuckDuckGo lookup. Failures are non-fatal: the node
logs and returns empty results so generation can still respond gracefully.
"""
from __future__ import annotations

import logging
import os

import httpx
from langchain_core.documents import Document

from orchestrator.state import RAGState

logger = logging.getLogger(__name__)

MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))


def _tavily_search(query: str, api_key: str) -> list[Document]:
    """Search via the Tavily API."""
    response = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": MAX_RESULTS},
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    return [
        Document(
            page_content=item.get("content", ""),
            metadata={"source": item.get("url", "web"), "type": "web"},
        )
        for item in results
    ]


def _duckduckgo_search(query: str) -> list[Document]:
    """Keyless fallback using the DuckDuckGo instant-answer API."""
    response = httpx.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    docs: list[Document] = []
    abstract = data.get("AbstractText")
    if abstract:
        docs.append(
            Document(
                page_content=abstract,
                metadata={"source": data.get("AbstractURL", "duckduckgo"), "type": "web"},
            )
        )
    for topic in data.get("RelatedTopics", [])[:MAX_RESULTS]:
        text = topic.get("Text")
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": topic.get("FirstURL", "duckduckgo"), "type": "web"},
                )
            )
    return docs[:MAX_RESULTS]


def web_search_fallback(state: RAGState) -> dict:
    """Fetch web context when retrieval/grading yielded nothing relevant.

    Returns:
        dict: Partial state with web ``documents`` (also promoted to
        ``graded_documents``) and ``web_search_triggered=True``.
    """
    query = state["question"]
    api_key = os.getenv("TAVILY_API_KEY", "")
    try:
        docs = _tavily_search(query, api_key) if api_key else _duckduckgo_search(query)
    except Exception as exc:
        logger.exception("Web search failed")
        return {"web_search_triggered": True, "error": str(exc), "source_types": ["web"]}

    return {
        "documents": docs,
        "graded_documents": docs,
        "web_search_triggered": True,
        "source_types": ["web"],
    }
