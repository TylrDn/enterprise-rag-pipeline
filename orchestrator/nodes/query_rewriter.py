"""Query rewriting node — improves retrieval recall."""
from langchain_openai import ChatOpenAI
from orchestrator.state import RAGState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL", "meta/llama3-70b-instruct"),
    base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
)

REWRITE_PROMPT = """You are a query rewriting expert. Rewrite the following question to maximize retrieval recall from a document store.
Make it more specific, expand acronyms, and rephrase ambiguities.
Return only the rewritten query, nothing else.

Original question: {question}"""


def rewrite_query(state: RAGState) -> RAGState:
    question = state["question"]
    retry = state.get("retry_count", 0)
    prompt = REWRITE_PROMPT.format(question=question)
    rewritten = llm.invoke(prompt).content.strip()
    return {
        **state,
        "rewritten_query": rewritten,
        "retry_count": retry,
    }
