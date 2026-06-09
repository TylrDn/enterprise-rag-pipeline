"""Tests for the RAGPipeline convenience orchestrator."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from orchestrator.pipeline import RAGPipeline
from pipeline.generator import GenerationResult
from pipeline.indexer import IndexResult


def _pipeline() -> RAGPipeline:
    indexer = MagicMock()
    indexer.index.return_value = IndexResult(total=1, indexed=1)
    generator = MagicMock()
    generator.generate.return_value = GenerationResult(answer="a", sources=["s"])
    return RAGPipeline(
        embedder=MagicMock(),
        backend=MagicMock(),
        indexer=indexer,
        generator=generator,
    )


def test_index_updates_corpus():
    pipe = _pipeline()
    result = pipe.index([Document(page_content="d")])
    assert result.indexed == 1
    assert len(pipe._corpus) == 1


def test_index_dry_run_leaves_corpus_empty():
    pipe = _pipeline()
    pipe.index([Document(page_content="d")], dry_run=True)
    assert pipe._corpus == []


def test_query_returns_generation_result():
    pipe = _pipeline()
    with patch("orchestrator.pipeline.HybridRetriever") as hybrid:
        retriever = MagicMock()
        retriever.retrieve.return_value = [Document(page_content="d")]
        hybrid.return_value = retriever
        result = pipe.query("question")
    assert result.answer == "a"


def test_stream_yields_tokens():
    pipe = _pipeline()
    pipe.generator.stream.return_value = iter(["t1", "t2"])
    with patch("orchestrator.pipeline.HybridRetriever") as hybrid:
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        hybrid.return_value = retriever
        tokens = list(pipe.stream("question"))
    assert tokens == ["t1", "t2"]
