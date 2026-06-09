"""Tests for ingestion loaders (web, Slack, SQL)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, text

from ingestion.slack_loader import SlackLoader
from ingestion.sql_loader import SQLLoader
from ingestion.web_loader import WebLoader


def test_web_loader_extracts_text():
    html = "<html><body><p>Hello world content for rag</p><script>x()</script></body></html>"
    resp = MagicMock()
    resp.text = html
    resp.raise_for_status.return_value = None
    with patch("ingestion.web_loader.httpx.get", return_value=resp):
        docs = WebLoader().load("http://example.com")
    assert len(docs) >= 1
    assert "Hello world content" in docs[0].page_content


def test_web_loader_failure_returns_empty():
    with patch("ingestion.web_loader.httpx.get", side_effect=Exception("net")):
        docs = WebLoader().load("http://bad.example")
    assert docs == []


def test_slack_loader_parses_messages(tmp_path):
    path = tmp_path / "general.json"
    path.write_text(
        json.dumps(
            [
                {"user": "u1", "ts": "123", "text": "hello team"},
                {"user": "u2", "ts": "124", "text": ""},
            ]
        )
    )
    docs = SlackLoader().load_file(path)
    assert len(docs) >= 1
    assert "hello team" in docs[0].page_content
    assert docs[0].metadata["source"] == "slack:general"


def test_sql_loader_reads_rows(tmp_path):
    url = f"sqlite:///{tmp_path / 'demo.db'}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE items (id INTEGER, name TEXT)"))
        conn.execute(text("INSERT INTO items VALUES (1, 'alpha'), (2, 'beta')"))

    loader = SQLLoader(db_url=url, rows_per_chunk=10)
    docs = loader.load_table("items")
    assert len(docs) == 1
    assert "name: alpha" in docs[0].page_content
    assert loader.load_all_tables()
