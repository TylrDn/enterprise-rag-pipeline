"""Tests for PDF loader."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@patch("ingestion.pdf_loader.fitz.open")
def test_pdf_loader_returns_documents(mock_fitz_open):
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "This is test content for the RAG pipeline."
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.__len__ = MagicMock(return_value=1)
    mock_fitz_open.return_value.__enter__ = MagicMock(return_value=mock_doc)
    mock_fitz_open.return_value = mock_doc

    from ingestion.pdf_loader import PDFLoader
    loader = PDFLoader(chunk_size=256)
    docs = loader.load("/fake/test.pdf")
    assert isinstance(docs, list)
