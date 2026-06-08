"""RAGAS evaluation harness — faithfulness, answer relevancy, context recall."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY = os.getenv("NVIDIA_API_KEY", "")


def build_ragas_dataset(qa_pairs: list[dict], pipeline) -> Dataset:
    """
    qa_pairs: [{"question": ..., "ground_truth": ...}, ...]
    pipeline: object with .generate_with_sources(query) -> {"answer": ..., "sources": [...]}
    """
    rows = []
    for pair in qa_pairs:
        q = pair["question"]
        result = pipeline.generate_with_sources(q)
        rows.append({
            "question": q,
            "answer": result["answer"],
            "contexts": result.get("contexts", [result["answer"]]),
            "ground_truth": pair["ground_truth"],
        })
    return Dataset.from_list(rows)


def run_ragas_eval(dataset: Dataset, output_path: str = "evals/reports/ragas_results.json") -> dict[str, float]:
    llm = ChatOpenAI(
        model="meta/llama-3.1-70b-instruct",
        openai_api_base=NIM_BASE_URL,
        openai_api_key=NIM_API_KEY,
        temperature=0.0,
    )
    embeddings = OpenAIEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        openai_api_base=NIM_BASE_URL,
        openai_api_key=NIM_API_KEY,
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_pandas().to_json(output_path, orient="records", indent=2)
    print(f"RAGAS results saved to {output_path}")
    print(result)
    return dict(result)


if __name__ == "__main__":
    from evals.datasets.sample_qa import QA_PAIRS
    from pipeline.generator import RAGGenerator
    # NOTE: Requires a running retriever — wire in your pipeline here
    print("Run via: python -m evals.ragas_eval after wiring a retriever.")
