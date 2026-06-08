"""RAG generation layer: retrieve → format context → generate answer."""
from __future__ import annotations

import os
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY = os.getenv("NVIDIA_API_KEY", "")
GEN_MODEL = os.getenv("GEN_MODEL", "meta/llama-3.1-70b-instruct")

RAG_PROMPT = ChatPromptTemplate.from_template("""\
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] (source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


class RAGGenerator:
    """Full RAG chain: retriever + NIM-backed LLM generation."""

    def __init__(self, retriever) -> None:
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=GEN_MODEL,
            openai_api_base=NIM_BASE_URL,
            openai_api_key=NIM_API_KEY,
            temperature=0.0,
        )
        self.chain = (
            {"context": self.retriever | _format_context, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def generate(self, query: str) -> str:
        return self.chain.invoke(query)

    def generate_with_sources(self, query: str) -> dict:
        docs = self.retriever.invoke(query)
        answer = self.chain.invoke(query)
        sources = list({doc.metadata.get("source", "unknown") for doc in docs})
        return {"answer": answer, "sources": sources}
