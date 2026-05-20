"""Retrieval evaluation helpers.

The default CLI path is intentionally retrieval-only:

1. context_precision — how many retrieved chunks include the expected answer?
2. hit rate — did at least one retrieved chunk contain the expected answer?
3. MRR — how early did the expected answer appear?

The optional RAGAS helper is available for later generated-answer evaluation, but it is
not part of the default benchmark run.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

# suppress the noisy ragas deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")


@dataclass
class EvalResult:
    context_precision: float
    answer_relevancy: float | None  # None if we skip generation eval
    faithfulness: float | None
    hit_rate: float  # fraction of queries where correct answer was in retrieved context
    mrr: float  # mean reciprocal rank


def evaluate_retrieval(
    questions: list[str],
    ground_truths: list[str],
    retrieved_contexts: list[list[str]],
    top_k: int = 5,
) -> EvalResult:
    """Evaluate retrieval quality without needing an LLM for generation.

    This is the practical evaluation — does the retriever surface the right chunks?
    We skip faithfulness/relevancy (those need a generator LLM) and focus on
    hit rate and MRR which are cheap to compute.
    """
    hit_count = 0
    reciprocal_ranks = []

    for gt, contexts in zip(ground_truths, retrieved_contexts):
        gt_lower = gt.lower().strip()
        found = False
        for rank, ctx in enumerate(contexts[:top_k], 1):
            # check if ground truth answer appears in the retrieved context
            if gt_lower in ctx.lower():
                if not found:
                    reciprocal_ranks.append(1.0 / rank)
                    found = True
                    hit_count += 1
        if not found:
            reciprocal_ranks.append(0.0)

    n = len(questions)
    hit_rate = hit_count / n if n > 0 else 0.0
    mrr = sum(reciprocal_ranks) / n if n > 0 else 0.0

    # context precision: what fraction of top-k chunks are actually relevant?
    # we approximate this by checking if the ground truth is contained
    precisions = []
    for gt, contexts in zip(ground_truths, retrieved_contexts):
        gt_lower = gt.lower().strip()
        relevant = sum(1 for ctx in contexts[:top_k] if gt_lower in ctx.lower())
        precisions.append(relevant / min(top_k, len(contexts)) if contexts else 0.0)

    context_precision = sum(precisions) / len(precisions) if precisions else 0.0

    return EvalResult(
        context_precision=round(context_precision, 4),
        answer_relevancy=None,
        faithfulness=None,
        hit_rate=round(hit_rate, 4),
        mrr=round(mrr, 4),
    )


def evaluate_with_ragas(
    questions: list[str],
    ground_truths: list[str],
    retrieved_contexts: list[list[str]],
    answers: list[str] | None = None,
) -> EvalResult:
    """Full RAGAS evaluation (needs OPENAI_API_KEY for the LLM judge).

    Only use this if you have an OpenAI key and want the full picture.
    For most benchmarking runs, evaluate_retrieval() is enough.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY required for RAGAS evaluation. "
            "Use evaluate_retrieval() for LLM-free evaluation."
        )

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation requires optional dependencies. "
            'Install with `pip install -e ".[ragas]"`.'
        ) from exc

    # if no generated answers, use ground truth as stand-in
    if answers is None:
        answers = ground_truths

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truths,
        }
    )

    result = evaluate(dataset, metrics=[context_precision, answer_relevancy, faithfulness])

    return EvalResult(
        context_precision=round(result["context_precision"], 4),
        answer_relevancy=round(result["answer_relevancy"], 4),
        faithfulness=round(result["faithfulness"], 4),
        hit_rate=0.0,  # not computed in RAGAS mode
        mrr=0.0,
    )
