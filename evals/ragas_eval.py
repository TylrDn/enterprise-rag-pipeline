"""RAGAS evaluation harness — faithfulness, answer relevancy, context recall.

Offline-only. Imports ``ragas``/``datasets`` at module load, so this module is
never imported on the main pipeline path (per repo conventions). Wire it to a
:class:`orchestrator.pipeline.RAGPipeline` via :func:`build_ragas_dataset`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY") or ""
DEFAULT_OUTPUT = "evals/results/ragas_results.json"

# Canonical RAGAS metrics, keyed by name for CLI selection.
METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_recall": context_recall,
    "context_precision": context_precision,
}


def build_ragas_dataset(qa_pairs: list[dict[str, str]], pipeline: Any) -> Dataset:
    """Build a RAGAS dataset by running the pipeline over ground-truth QA pairs.

    Args:
        qa_pairs: Items shaped ``{"question": ..., "ground_truth": ...}``.
        pipeline: An object exposing ``answer_with_contexts(question) ->
            {"answer": str, "contexts": list[str]}`` (e.g. ``RAGPipeline``).

    Returns:
        Dataset: A HuggingFace dataset with RAGAS-compatible columns.
    """
    rows: list[dict[str, Any]] = []
    for pair in qa_pairs:
        question = pair["question"]
        result = pipeline.answer_with_contexts(question)
        contexts = result.get("contexts") or [result["answer"]]
        rows.append(
            {
                "question": question,
                "answer": result["answer"],
                "contexts": contexts,
                "ground_truth": pair["ground_truth"],
            }
        )
        logger.info("Generated answer for: %s", question)
    return Dataset.from_list(rows)


def run_ragas_eval(
    dataset: Dataset,
    output_path: str = DEFAULT_OUTPUT,
    metric_names: list[str] | None = None,
) -> dict[str, float]:
    """Run RAGAS over ``dataset`` using NIM-backed LLM and embeddings.

    Args:
        dataset: A dataset built by :func:`build_ragas_dataset`.
        output_path: Where to write the per-sample JSON results.
        metric_names: Subset of :data:`METRICS` to evaluate (defaults to all).

    Returns:
        dict[str, float]: Aggregated metric scores.
    """
    selected = [METRICS[name] for name in (metric_names or list(METRICS))]

    llm = ChatOpenAI(
        model=os.getenv("CHAT_MODEL", "meta/llama-3.1-70b-instruct"),
        base_url=NIM_BASE_URL,
        api_key=NIM_API_KEY,  # type: ignore[arg-type]  # str coerced to SecretStr
        temperature=0.0,
    )
    embeddings = OpenAIEmbeddings(
        model=os.getenv("NIM_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"),
        base_url=NIM_BASE_URL,
        api_key=NIM_API_KEY,  # type: ignore[arg-type]  # str coerced to SecretStr
    )

    result = evaluate(dataset, metrics=selected, llm=llm, embeddings=embeddings)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_pandas().to_json(output_path, orient="records", indent=2)
    logger.info("RAGAS results saved to %s", output_path)
    logger.info("RAGAS scores: %s", result)
    return dict(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Run via evals/ragas_runner.py to wire a dataset and pipeline.")
