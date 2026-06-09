"""Unit tests for pipeline/retriever.py.

All BM25, vector store, and cross-encoder calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from pipeline.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORPUS = [
    Document(page_content="NVIDIA NIM provides optimised inference endpoints.", metadata={"source": "a.pdf"}),  # noqa: E501
    Document(page_content="pgvector stores dense embeddings in PostgreSQL.", metadata={"source": "b.pdf"}),  # noqa: E501
    Document(page_content="LangChain supports EnsembleRetriever for hybrid search.", metadata={"source": "c.pdf"}),  # noqa: E501
    Document(page_content="Cross-encoder models rerank candidate documents.", metadata={"source": "d.pdf"}),  # noqa: E501
    Document(page_content="RAG pipelines combine retrieval and generation.", metadata={"source": "e.pdf"}),  # noqa: E501
]

RETURNED_DOCS = [
    Document(page_content="Relevant result 1", metadata={"source": "a.pdf"}),
    Document(page_content="Relevant result 2", metadata={"source": "b.pdf"}),
]


@pytest.fixture()
def mock_vector_store():
    vs = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = RETURNED_DOCS
    vs.as_retriever.return_value = mock_retriever
    return vs


@pytest.fixture()
def retriever_no_rerank(mock_vector_store):
    return HybridRetriever(
        vector_store=mock_vector_store,
        corpus_documents=CORPUS,
        top_k=5,
        rerank=False,
        hybrid_alpha=0.6,
    )


@pytest.fixture()
def retriever_with_rerank(mock_vector_store):
    return HybridRetriever(
        vector_store=mock_vector_store,
        corpus_documents=CORPUS,
        top_k=5,
        rerank=True,
        rerank_top_n=2,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )


# ---------------------------------------------------------------------------
# Tests — no rerank path
# ---------------------------------------------------------------------------

class TestHybridRetrieverNoRerank:
    def test_retrieve_returns_list(self, retriever_no_rerank):
        with patch(
            "pipeline.retriever.BM25Retriever.from_documents",
        ) as mock_bm25_cls, patch(
            "pipeline.retriever.EnsembleRetriever"
        ) as mock_ensemble_cls:
            mock_bm25 = MagicMock()
            mock_bm25.k = 5
            mock_bm25_cls.return_value = mock_bm25

            mock_ensemble = MagicMock()
            mock_ensemble.invoke.return_value = RETURNED_DOCS
            mock_ensemble_cls.return_value = mock_ensemble

            results = retriever_no_rerank.retrieve("What is NVIDIA NIM?")

        assert isinstance(results, list)
        assert len(results) == 2

    def test_retrieve_calls_ensemble_invoke(self, retriever_no_rerank):
        with patch("pipeline.retriever.BM25Retriever.from_documents") as mock_bm25_cls, \
             patch("pipeline.retriever.EnsembleRetriever") as mock_ensemble_cls:
            mock_bm25_cls.return_value = MagicMock(k=5)
            mock_ensemble = MagicMock()
            mock_ensemble.invoke.return_value = RETURNED_DOCS
            mock_ensemble_cls.return_value = mock_ensemble

            retriever_no_rerank.retrieve("pgvector")
            mock_ensemble.invoke.assert_called_once_with("pgvector")

    def test_weights_sum_to_one(self, retriever_no_rerank):
        alpha = retriever_no_rerank.hybrid_alpha
        assert abs(alpha + (1.0 - alpha) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Tests — rerank path
# ---------------------------------------------------------------------------

class TestHybridRetrieverWithRerank:
    def test_rerank_pipeline_invoked(self, retriever_with_rerank):
        with patch("pipeline.retriever.BM25Retriever.from_documents") as mock_bm25_cls, \
             patch("pipeline.retriever.EnsembleRetriever") as mock_ensemble_cls, \
             patch("pipeline.retriever.HuggingFaceCrossEncoder") as mock_ce_cls, \
             patch("pipeline.retriever.CrossEncoderReranker") as mock_reranker_cls, \
             patch("pipeline.retriever.ContextualCompressionRetriever") as mock_ccr_cls:

            mock_bm25_cls.return_value = MagicMock(k=5)
            mock_ensemble_cls.return_value = MagicMock()
            mock_ce_cls.return_value = MagicMock()
            mock_reranker_cls.return_value = MagicMock()

            mock_ccr = MagicMock()
            mock_ccr.invoke.return_value = RETURNED_DOCS[:1]
            mock_ccr_cls.return_value = mock_ccr

            results = retriever_with_rerank.retrieve("reranking test")

        mock_ce_cls.assert_called_once_with(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        mock_reranker_cls.assert_called_once()
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Tests — retrieve_with_sources
# ---------------------------------------------------------------------------

def test_retrieve_with_sources_structure(retriever_no_rerank):
    with patch("pipeline.retriever.BM25Retriever.from_documents") as mock_bm25_cls, \
         patch("pipeline.retriever.EnsembleRetriever") as mock_ensemble_cls:
        mock_bm25_cls.return_value = MagicMock(k=5)
        mock_ensemble = MagicMock()
        mock_ensemble.invoke.return_value = RETURNED_DOCS
        mock_ensemble_cls.return_value = mock_ensemble

        results = retriever_no_rerank.retrieve_with_sources("test")

    assert all("content" in r and "source" in r and "metadata" in r for r in results)


# ---------------------------------------------------------------------------
# Tests — update_corpus
# ---------------------------------------------------------------------------

def test_update_corpus_resets_retriever(retriever_no_rerank):
    retriever_no_rerank._retriever = MagicMock()  # simulate built state
    new_docs = [Document(page_content="New doc", metadata={})]
    retriever_no_rerank.update_corpus(new_docs)
    assert retriever_no_rerank._retriever is None
    assert retriever_no_rerank._corpus_documents == new_docs


# ---------------------------------------------------------------------------
# Tests — as_langchain_retriever
# ---------------------------------------------------------------------------

def test_as_langchain_retriever_returns_retriever(retriever_no_rerank):
    with patch("pipeline.retriever.BM25Retriever.from_documents") as mock_bm25_cls, \
         patch("pipeline.retriever.EnsembleRetriever") as mock_ensemble_cls:
        mock_bm25_cls.return_value = MagicMock(k=5)
        mock_ensemble_cls.return_value = MagicMock()
        lc_retriever = retriever_no_rerank.as_langchain_retriever()
    assert lc_retriever is not None
