"""CLI runner for RAGAS evaluation against the enterprise RAG pipeline."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from evals.datasets.sample_qa import QA_PAIRS
from evals.ragas_eval import METRICS, build_ragas_dataset, run_ragas_eval
from orchestrator.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> list[dict[str, str]]:
    """Load QA pairs from ``.json`` (list) or ``.jsonl`` (newline-delimited).

    Args:
        path: Path to a dataset file.

    Returns:
        list[dict[str, str]]: Items with ``question`` and ``ground_truth`` keys.

    Raises:
        ValueError: If the file format is unsupported or malformed.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "question": item["question"],
                    "ground_truth": item.get("ground_truth", item.get("answer", "")),
                }
            )
        return rows

    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return [
                {
                    "question": item["question"],
                    "ground_truth": item.get("ground_truth", item.get("answer", "")),
                }
                for item in data
            ]
        raise ValueError("JSON dataset must be a list of objects.")

    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for the RAG pipeline.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to .json or .jsonl QA dataset (defaults to built-in sample_qa).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("evals/results/ragas_results.json"),
        help="Where to write per-sample RAGAS results.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=list(METRICS.keys()),
        default=list(METRICS.keys()),
        help="RAGAS metrics to compute.",
    )
    return parser.parse_args()


def main() -> None:
    """Run RAGAS evaluation and write results to disk."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    qa_pairs = load_dataset(args.dataset) if args.dataset else QA_PAIRS
    pipeline = RAGPipeline()
    dataset = build_ragas_dataset(qa_pairs, pipeline)
    scores = run_ragas_eval(
        dataset,
        output_path=str(args.output_file),
        metric_names=args.metrics,
    )
    logger.info("Aggregated RAGAS scores: %s", scores)


if __name__ == "__main__":
    main()
