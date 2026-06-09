"""Schema-aware SQL table → Document converter."""
from __future__ import annotations

import os

from langchain_core.documents import Document
from sqlalchemy import create_engine, inspect, text

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./demo.db")


class SQLLoader:
    """Convert SQL tables into LangChain Documents for RAG ingestion."""

    def __init__(self, db_url: str = DB_URL, rows_per_chunk: int = 50) -> None:
        self.engine = create_engine(db_url)
        self.rows_per_chunk = rows_per_chunk

    def _get_schema(self, table_name: str) -> str:
        insp = inspect(self.engine)
        cols = insp.get_columns(table_name)
        return ", ".join(f"{c['name']} ({c['type']})" for c in cols)

    def load_table(self, table_name: str) -> list[Document]:
        schema = self._get_schema(table_name)
        docs: list[Document] = []
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table_name}"))
            keys = list(result.keys())
            batch: list[str] = []
            chunk_idx = 0
            for row in result:
                row_text = " | ".join(f"{k}: {v}" for k, v in zip(keys, row))
                batch.append(row_text)
                if len(batch) >= self.rows_per_chunk:
                    docs.append(Document(
                        page_content="\n".join(batch),
                        metadata={
                            "source": f"sql:{table_name}",
                            "chunk": chunk_idx,
                            "schema": schema,
                        },
                    ))
                    batch = []
                    chunk_idx += 1
            if batch:
                docs.append(Document(
                    page_content="\n".join(batch),
                    metadata={"source": f"sql:{table_name}", "chunk": chunk_idx, "schema": schema},
                ))
        return docs

    def load_all_tables(self) -> list[Document]:
        insp = inspect(self.engine)
        docs: list[Document] = []
        for table in insp.get_table_names():
            docs.extend(self.load_table(table))
        return docs
