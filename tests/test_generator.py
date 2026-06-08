"""Unit tests for pipeline/generator.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from pipeline.generator import (
    Generator,
    GenerationResult,
    _format_context,
    _parse_grounded_flag,
    _NO_CONTEXT_REPLY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DOCS = [
    Document(page_content="NVIDIA NIM accelerates inference.", metadata={"source": "nim.pdf"}),
    Document(page_content="pgvector enables vector search in Postgres.", metadata={"source": "pg.pdf"}),
]


@pytest.fixture()
def generator(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    return Generator(
        model="meta/llama3-70b-instruct",
        temperature=0.0,
        hallucination_check=False,
    )


@pytest.fixture()
def generator_with_hcheck(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    return Generator(
        model="meta/llama3-70b-instruct",
        temperature=0.0,
        hallucination_check=True,
    )


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------

def test_format_context_includes_source():
    ctx = _format_context(DOCS)
    assert "nim.pdf" in ctx
    assert "NVIDIA NIM" in ctx


def test_format_context_numbered():
    ctx = _format_context(DOCS)
    assert "[1]" in ctx
    assert "[2]" in ctx


def test_format_context_empty():
    assert _format_context([]) == ""


# ---------------------------------------------------------------------------
# _parse_grounded_flag
# ---------------------------------------------------------------------------

def test_parse_grounded_true():
    assert _parse_grounded_flag('{"grounded": true}') is True


def test_parse_grounded_false():
    assert _parse_grounded_flag('{"grounded": false}') is False


def test_parse_grounded_malformed_defaults_true():
    assert _parse_grounded_flag("not json at all") is True


# ---------------------------------------------------------------------------
# Generator.generate()
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_generate_returns_result(self, generator):
        with patch("pipeline.generator.ChatNVIDIA") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.streaming = False
            mock_chain_result = "NVIDIA NIM provides fast inference."
            # Simulate chain.invoke
            mock_llm.__or__ = MagicMock(return_value=mock_llm)
            mock_llm_cls.return_value = mock_llm

            with patch("pipeline.generator.RAG_PROMPT") as mock_prompt:
                mock_chain = MagicMock()
                mock_chain.invoke.return_value = mock_chain_result
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                mock_chain.__or__ = MagicMock(return_value=mock_chain)

                result = generator.generate("What is NIM?", DOCS)

        assert isinstance(result, GenerationResult)

    def test_generate_refused_on_empty_docs(self, generator):
        result = generator.generate("Any question", [])
        assert result.refused is True
        assert result.answer == _NO_CONTEXT_REPLY

    def test_generate_refused_on_blank_docs(self, generator):
        blank_docs = [Document(page_content="   ", metadata={})]
        result = generator.generate("Any question", blank_docs)
        assert result.refused is True

    def test_sources_extracted(self, generator):
        with patch("pipeline.generator.ChatNVIDIA") as mock_llm_cls, \
             patch("pipeline.generator.RAG_PROMPT") as mock_prompt:
            mock_llm = MagicMock(streaming=False)
            mock_llm_cls.return_value = mock_llm
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "An answer."
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.__or__ = MagicMock(return_value=mock_chain)

            result = generator.generate("question", DOCS)

        assert "nim.pdf" in result.sources or "pg.pdf" in result.sources


# ---------------------------------------------------------------------------
# Generator.generate() with hallucination check
# ---------------------------------------------------------------------------

class TestHallucinationCheck:
    def test_grounded_true_propagates(self, generator_with_hcheck):
        with patch("pipeline.generator.ChatNVIDIA") as mock_llm_cls, \
             patch("pipeline.generator.RAG_PROMPT") as mock_prompt, \
             patch.object(generator_with_hcheck, "_check_grounding", return_value=True):
            mock_llm = MagicMock(streaming=False)
            mock_llm_cls.return_value = mock_llm
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Grounded answer."
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.__or__ = MagicMock(return_value=mock_chain)

            result = generator_with_hcheck.generate("q", DOCS)
        assert result.grounded is True

    def test_grounded_false_propagates(self, generator_with_hcheck):
        with patch("pipeline.generator.ChatNVIDIA") as mock_llm_cls, \
             patch("pipeline.generator.RAG_PROMPT") as mock_prompt, \
             patch.object(generator_with_hcheck, "_check_grounding", return_value=False):
            mock_llm = MagicMock(streaming=False)
            mock_llm_cls.return_value = mock_llm
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Possibly hallucinated."
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.__or__ = MagicMock(return_value=mock_chain)

            result = generator_with_hcheck.generate("q", DOCS)
        assert result.grounded is False


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------

def test_result_str_refused():
    r = GenerationResult(answer="I don't know.", refused=True)
    assert "REFUSED" in str(r)


def test_result_str_ungrounded():
    r = GenerationResult(answer="Some answer.", grounded=False)
    assert "UNGROUNDED" in str(r)


def test_result_str_normal():
    r = GenerationResult(answer="Normal answer.")
    assert "REFUSED" not in str(r)
    assert "UNGROUNDED" not in str(r)
