"""Unit tests for NIM embedder."""
from unittest.mock import MagicMock, patch
from embeddings.nim_embedder import NIMEmbedder


def test_embed_query_returns_list():
    with patch("embeddings.nim_embedder.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1024)]
        mock_client.embeddings.create.return_value = mock_response
        MockOpenAI.return_value = mock_client
        embedder = NIMEmbedder()
        result = embedder.embed_query("What is RAG?")
        assert isinstance(result, list)
        assert len(result) == 1024


def test_embed_documents_batches():
    with patch("embeddings.nim_embedder.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1024) for _ in range(3)]
        mock_client.embeddings.create.return_value = mock_response
        MockOpenAI.return_value = mock_client
        embedder = NIMEmbedder(batch_size=32)
        results = embedder.embed_documents(["doc1", "doc2", "doc3"])
        assert len(results) == 3
