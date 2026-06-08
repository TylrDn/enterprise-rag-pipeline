"""RAGAS-based evaluation harness for the RAG pipeline.

Measures:
  - faithfulness      : are answers grounded in retrieved context?
  - answer_relevancy  : does the answer address the question?
  - context_precision : are retrieved chunks relevant to the question?
  - context_recall    : are all necessary chunks retrieved?

Usage::

    from evals.ragas_eval import run_eval
    results = run_eval(pipeline, eval_dataset)
    print(results.to_dict())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample."""
    question: str
    ground_truth: str
    reference_contexts: List[str]


@dataclass
class EvalResult:
    """Aggregated RAGAS evaluation results."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "n_samples": self.n_samples,
        }

    def __str__(self) -> str:
        d = self.to_dict()
        return (
            f"EvalResult(n={d['n_samples']}) | "
            f"faithfulness={d['faithfulness']} | "
            f"answer_relevancy={d['answer_relevancy']} | "
            f"context_precision={d['context_precision']} | "
            f"context_recall={d['context_recall']}"
        )


def run_eval(
    pipeline,
    samples: List[EvalSample],
    langfuse_trace: bool = False,
) -> EvalResult:
    """Run RAGAS metrics over a list of EvalSamples.

    Args:
        pipeline:       A RAGPipeline instance.
        samples:        List of EvalSample with questions, ground truth, and contexts.
        langfuse_trace: If True, log each sample to Langfuse for inspection.

    Returns:
        EvalResult with averaged metric scores.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            "ragas and datasets are required for evaluation. "
            "Install with: pip install ragas datasets"
        ) from e

    rows = []
    for sample in samples:
        result = pipeline.query(sample.question)
        rows.append({
            "question": sample.question,
            "answer": result.answer,
            "contexts": sample.reference_contexts,
            "ground_truth": sample.ground_truth,
        })
        if langfuse_trace:
            logger.info(
                "[RAGAS] question=%s | grounded=%s",
                sample.question[:60],
                result.grounded,
            )

    dataset = Dataset.from_list(rows)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    return EvalResult(
        faithfulness=scores["faithfulness"],
        answer_relevancy=scores["answer_relevancy"],
        context_precision=scores["context_precision"],
        context_recall=scores["context_recall"],
        n_samples=len(samples),
    )
