"""Unit tests for FastAPI RAG gateway."""
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint():
    mock_result = {
        "question": "What is RAG?",
        "rewritten_query": "What is Retrieval Augmented Generation?",
        "documents": [],
        "graded_documents": [],
        "generation": "RAG stands for Retrieval Augmented Generation.",
        "hallucination_score": 0.9,
        "answer_grade": "supported",
        "retry_count": 0,
        "source_types": [],
    }
    with patch("api.main.rag_graph") as mock_graph:
        mock_graph.invoke.return_value = mock_result
        response = client.post("/query", json={"question": "What is RAG?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer_grade"] == "supported"
        assert data["hallucination_score"] == 0.9
