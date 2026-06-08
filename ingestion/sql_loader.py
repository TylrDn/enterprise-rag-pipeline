"""SQL table ingestion — converts rows to documents."""
from sqlalchemy import create_engine, text
from langchain_core.documents import Document
from vectorstore.pgvector_store import PGVectorStore
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


def load_sql_table(table: str, columns: List[str], db_url: str = None) -> List[Document]:
    url = db_url or os.getenv("DATABASE_URL", "").replace("+asyncpg", "")
    engine = create_engine(url)
    col_str = ", ".join(columns)
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT {col_str} FROM {table}")).fetchall()
    docs = []
    for row in rows:
        content = " | ".join(f"{col}: {val}" for col, val in zip(columns, row))
        docs.append(Document(
            page_content=content,
            metadata={"source": table, "type": "sql"},
        ))
    return docs


def ingest_sql_table(table: str, columns: List[str], store: PGVectorStore = None) -> List[str]:
    docs = load_sql_table(table, columns)
    vs = store or PGVectorStore()
    ids = vs.add_documents(docs)
    print(f"Ingested {len(docs)} rows from {table}")
    return ids
