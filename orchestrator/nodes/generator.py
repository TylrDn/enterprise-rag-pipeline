"""Answer generation node using NIM."""
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from orchestrator.state import RAGState
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL", "meta/llama3-70b-instruct"),
    base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.1,
    max_tokens=1024,
)

GENERATE_PROMPT = """You are a knowledgeable assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Cite the source of each fact using [source: <source>] notation.

Context:
{context}

Question: {question}

Answer:"""


def format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def generate_answer(state: RAGState) -> RAGState:
    docs = state.get("graded_documents") or state.get("documents", [])
    question = state.get("rewritten_query") or state["question"]
    context = format_context(docs)
    prompt = GENERATE_PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt).content.strip()
    return {**state, "generation": answer}
