"""LLM generation layer — NIM-backed with Langfuse tracing."""
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from typing import List
import os


def get_llm(model: str = "meta/llama-3.1-70b-instruct", temperature: float = 0.0) -> ChatOpenAI:
    """Instantiate a NIM-backed ChatOpenAI LLM with Langfuse tracing."""
    langfuse_handler = CallbackHandler(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ["NVIDIA_API_KEY"],
        openai_api_base=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        temperature=temperature,
        callbacks=[langfuse_handler],
    )


def generate_answer(question: str, context_docs: List[str], llm: ChatOpenAI | None = None) -> str:
    """Generate an answer grounded in the provided context documents."""
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser

    if llm is None:
        llm = get_llm()

    context = "\n\n---\n\n".join(context_docs)
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the following context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
