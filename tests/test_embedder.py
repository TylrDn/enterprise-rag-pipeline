"""Tests for Embedder wrapper."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("pipeline.embedder.OpenAIEmbeddings")
def test_embedder_init(mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    from pipeline.embedder import Embedder
    e = Embedder(model="nvidia/nv-embedqa-e5-v5")
    assert e.model == "nvidia/nv-embedqa-e5-v5"
