"""Tests for the CRAG web-search fallback node."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from orchestrator.nodes.web_search import web_search_fallback
from orchestrator.state import initial_state


def _make_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_duckduckgo_path(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    payload = {
        "AbstractText": "NIM serves models.",
        "AbstractURL": "http://example.com",
        "RelatedTopics": [{"Text": "Related", "FirstURL": "http://example.com/r"}],
    }
    with patch(
        "orchestrator.nodes.web_search.httpx.get", return_value=_make_response(payload)
    ):
        out = web_search_fallback(initial_state("what is nim?"))
    assert out["web_search_triggered"] is True
    assert len(out["documents"]) >= 1


def test_tavily_path(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    payload = {"results": [{"content": "Tavily content", "url": "http://t.example"}]}
    with patch(
        "orchestrator.nodes.web_search.httpx.post", return_value=_make_response(payload)
    ):
        out = web_search_fallback(initial_state("query"))
    assert out["documents"][0].metadata["source"] == "http://t.example"


def test_failure_is_non_fatal(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with patch(
        "orchestrator.nodes.web_search.httpx.get", side_effect=Exception("network down")
    ):
        out = web_search_fallback(initial_state("query"))
    assert out["web_search_triggered"] is True
    assert "error" in out
