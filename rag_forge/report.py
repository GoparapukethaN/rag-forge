"""Generate benchmark reports: markdown, JSON, and Pareto plot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_forge.bench import RunResult


def generate_markdown_report(results: list[RunResult], output_path: str | None = None) -> str:
    """Generate a markdown table of benchmark results, sorted by hit rate."""
    lines = [
        "# RAG-Forge Benchmark Results\n",
        f"Tested **{len(results)} configurations**\n",
        "| # | Chunker | Embedder | Retriever | Reranker | Hit Rate | MRR | Cached Query Latency |",
        "|---|---------|----------|-----------|----------|----------|-----|---------|",
    ]

    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r.chunker} | {r.embedder} | {r.retriever} "
            f"| {r.reranker} | {r.eval.hit_rate:.3f} "
            f"| {r.eval.mrr:.3f} | {r.latency_ms:.0f}ms |"
        )

    if results:
        best = results[0]
        fastest = min(results, key=lambda r: r.latency_ms)
        hit_delta = best.eval.hit_rate - fastest.eval.hit_rate
        latency_delta = best.latency_ms - fastest.latency_ms
        lines.extend([
            "",
            "## Recommendation",
            (
                f"Use `{best.config_id}` as the current default candidate. It has the "
                f"highest hit rate ({best.eval.hit_rate:.3f}) and MRR ({best.eval.mrr:.3f}) "
                f"in this run."
            ),
            "",
            (
                f"Compared with the fastest configuration (`{fastest.config_id}`), it changes "
                f"hit rate by {hit_delta:+.3f} and latency by {latency_delta:+.0f}ms."
            ),
            "",
            "## Best Configuration",
            f"- **Chunker:** {best.chunker}",
            f"- **Embedder:** {best.embedder}",
            f"- **Retriever:** {best.retriever}",
            f"- **Reranker:** {best.reranker}",
            f"- **Hit Rate:** {best.eval.hit_rate:.3f}",
            f"- **MRR:** {best.eval.mrr:.3f}",
            f"- **Context Precision:** {best.eval.context_precision:.3f}",
            f"- **Avg Cached Query Latency:** {best.latency_ms:.0f}ms",
            f"- **Chunks:** {best.num_chunks}",
        ])

    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")

    return report


def generate_json_report(
    results: list[RunResult],
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate a machine-readable benchmark report."""
    payload: dict[str, Any] = {
        "configuration_count": len(results),
        "recommendation": _recommendation_payload(results),
        "results": [_result_payload(result, rank) for rank, result in enumerate(results, 1)],
    }

    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def _recommendation_payload(results: list[RunResult]) -> dict[str, Any] | None:
    if not results:
        return None
    best = results[0]
    fastest = min(results, key=lambda result: result.latency_ms)
    return {
        "config_id": best.config_id,
        "reason": "highest_hit_rate_then_mrr",
        "hit_rate": best.eval.hit_rate,
        "mrr": best.eval.mrr,
        "context_precision": best.eval.context_precision,
        "latency_ms": best.latency_ms,
        "num_chunks": best.num_chunks,
        "fastest_config_id": fastest.config_id,
        "hit_rate_delta_vs_fastest": round(best.eval.hit_rate - fastest.eval.hit_rate, 4),
        "latency_delta_ms_vs_fastest": round(best.latency_ms - fastest.latency_ms, 1),
    }


def _result_payload(result: RunResult, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "config_id": result.config_id,
        "chunker": result.chunker,
        "embedder": result.embedder,
        "retriever": result.retriever,
        "reranker": result.reranker,
        "hit_rate": result.eval.hit_rate,
        "mrr": result.eval.mrr,
        "context_precision": result.eval.context_precision,
        "latency_ms": result.latency_ms,
        "num_chunks": result.num_chunks,
    }


def generate_pareto_plot(results: list[RunResult], output_path: str = "pareto.png"):
    """Scatter plot: hit rate vs latency. Pareto-optimal configs highlighted.

    The point is to show the tradeoff: you can get better accuracy but it costs latency.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    hit_rates = [r.eval.hit_rate for r in results]
    latencies = [r.latency_ms for r in results]
    labels = [f"{r.chunker}\n{r.embedder}\n{r.retriever}" for r in results]

    # find pareto-optimal points (highest hit rate for a given latency)
    pareto_mask = _pareto_front(hit_rates, latencies)

    fig, ax = plt.subplots(figsize=(10, 6))

    # non-pareto points
    non_pareto_x = [latencies[i] for i in range(len(results)) if not pareto_mask[i]]
    non_pareto_y = [hit_rates[i] for i in range(len(results)) if not pareto_mask[i]]
    ax.scatter(non_pareto_x, non_pareto_y, alpha=0.4, color="gray", s=40, label="Other configs")

    # pareto-optimal points
    pareto_x = [latencies[i] for i in range(len(results)) if pareto_mask[i]]
    pareto_y = [hit_rates[i] for i in range(len(results)) if pareto_mask[i]]
    ax.scatter(pareto_x, pareto_y, color="#e63946", s=80, zorder=5, label="Pareto-optimal")

    # label pareto points
    for i, idx in enumerate([j for j in range(len(results)) if pareto_mask[j]]):
        ax.annotate(
            labels[idx],
            (latencies[idx], hit_rates[idx]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=7,
            alpha=0.8,
        )

    ax.set_xlabel("Avg Latency per Query (ms)", fontsize=11)
    ax.set_ylabel("Hit Rate", fontsize=11)
    ax.set_title("RAG Configuration: Quality vs Latency Tradeoff", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _pareto_front(hit_rates: list[float], latencies: list[float]) -> list[bool]:
    """Find pareto-optimal points: maximize hit_rate, minimize latency."""
    n = len(hit_rates)
    is_pareto = [True] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has better or equal hit rate AND better or equal latency
            if hit_rates[j] >= hit_rates[i] and latencies[j] <= latencies[i]:
                if hit_rates[j] > hit_rates[i] or latencies[j] < latencies[i]:
                    is_pareto[i] = False
                    break

    return is_pareto
