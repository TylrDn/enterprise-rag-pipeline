"""Slack export JSON → LangChain Documents."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SlackLoader:
    """Load Slack export JSON files into LangChain Documents."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_file(self, path: str | Path) -> list[Document]:
        path = Path(path)
        with open(path) as f:
            messages = json.load(f)
        texts: list[str] = []
        for msg in messages:
            user = msg.get("user", "unknown")
            ts = msg.get("ts", "")
            text = msg.get("text", "").strip()
            if text:
                texts.append(f"[{ts}] {user}: {text}")
        channel = path.stem
        raw = Document(
            page_content="\n".join(texts),
            metadata={"source": f"slack:{channel}", "file": str(path)},
        )
        return self.splitter.split_documents([raw])

    def load_export(self, export_dir: str | Path) -> list[Document]:
        export_dir = Path(export_dir)
        docs: list[Document] = []
        for jf in sorted(export_dir.glob("**/*.json")):
            docs.extend(self.load_file(jf))
        return docs
