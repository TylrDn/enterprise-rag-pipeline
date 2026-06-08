"""Slack export JSON → LangChain Documents."""
import json
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import logging

logger = logging.getLogger(__name__)


def load_slack_export(export_dir: str | Path, channels: List[str] | None = None) -> List[Document]:
    """Parse a Slack data export directory into Documents."""
    export_path = Path(export_dir)
    docs: List[Document] = []
    channel_dirs = [d for d in export_path.iterdir() if d.is_dir()]
    if channels:
        channel_dirs = [d for d in channel_dirs if d.name in channels]

    for channel in channel_dirs:
        for json_file in sorted(channel.glob("*.json")):
            messages = json.loads(json_file.read_text())
            for msg in messages:
                if msg.get("type") == "message" and "text" in msg:
                    docs.append(Document(
                        page_content=msg["text"],
                        metadata={
                            "source": f"slack/{channel.name}/{json_file.name}",
                            "channel": channel.name,
                            "ts": msg.get("ts", ""),
                            "user": msg.get("user", "unknown"),
                            "loader": "slack",
                        }
                    ))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    logger.info(f"slack_loader: {len(chunks)} chunks from {export_path}")
    return chunks
