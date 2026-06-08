"""Slack channel ingestion via slack-sdk."""
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain_core.documents import Document
from embeddings.chunker import recursive_chunk
from vectorstore.pgvector_store import PGVectorStore
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class SlackLoader:
    def __init__(self):
        self.client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

    def load_channel(self, channel_id: str, limit: int = 200) -> List[Document]:
        try:
            result = self.client.conversations_history(channel=channel_id, limit=limit)
            messages = result.get("messages", [])
        except SlackApiError as e:
            print(f"Slack error: {e.response['error']}")
            return []

        docs = []
        for msg in messages:
            if msg.get("type") == "message" and not msg.get("subtype"):
                docs.append(Document(
                    page_content=msg.get("text", ""),
                    metadata={
                        "source": f"slack:{channel_id}",
                        "type": "slack",
                        "ts": msg.get("ts"),
                        "user": msg.get("user", "unknown"),
                    },
                ))
        return docs

    def ingest_channel(self, channel_id: str, store: PGVectorStore = None) -> List[str]:
        docs = self.load_channel(channel_id)
        if not docs:
            return []
        chunks = recursive_chunk(docs)
        vs = store or PGVectorStore()
        ids = vs.add_documents(chunks)
        print(f"Ingested {len(chunks)} chunks from slack:{channel_id}")
        return ids
