"""Schema-aware SQL table → LangChain Document converter."""
from sqlalchemy import create_engine, inspect, text
from langchain.schema import Document
from typing import List
import logging

logger = logging.getLogger(__name__)


def load_table(db_url: str, table_name: str, columns: List[str] | None = None) -> List[Document]:
    """Convert each row of a SQL table into a Document with schema metadata."""
    engine = create_engine(db_url)
    inspector = inspect(engine)
    schema_info = {col["name"]: col["type"] for col in inspector.get_columns(table_name)}

    col_clause = ", ".join(columns) if columns else "*"
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT {col_clause} FROM {table_name}")).mappings().all()

    docs: List[Document] = []
    for i, row in enumerate(rows):
        content = "\n".join(f"{k}: {v}" for k, v in row.items())
        docs.append(Document(
            page_content=content,
            metadata={
                "source": f"{db_url}/{table_name}",
                "row_index": i,
                "table": table_name,
                "schema": str(schema_info),
                "loader": "sql",
            }
        ))
    logger.info(f"sql_loader: {len(docs)} docs from {table_name}")
    return docs
