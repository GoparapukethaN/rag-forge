from pathlib import Path

from rag_forge.bench import RunResult
from rag_forge.evaluate import EvalResult
from rag_forge.report import generate_json_report, generate_markdown_report


def _result(
    chunker: str,
    embedder: str,
    retriever: str,
    hit_rate: float,
    mrr: float,
    latency_ms: float,
) -> RunResult:
    return RunResult(
        chunker=chunker,
        embedder=embedder,
        retriever=retriever,
        reranker="none",
        eval=EvalResult(
            context_precision=0.5,
            answer_relevancy=None,
            faithfulness=None,
            hit_rate=hit_rate,
            mrr=mrr,
        ),
        latency_ms=latency_ms,
        num_chunks=12,
    )


def test_markdown_report_includes_recommendation() -> None:
    results = [
        _result("recursive_512", "bge-small", "dense", 0.8, 0.7, 30),
        _result("fixed_512", "bge-small", "bm25", 0.7, 0.6, 10),
    ]

    report = generate_markdown_report(results)

    assert "## Recommendation" in report
    assert "`recursive_512|bge-small|dense|none`" in report
    assert "fastest configuration" in report
    assert "**Context Precision:** 0.500" in report


def test_json_report_writes_ranked_results(tmp_path: Path) -> None:
    output_path = tmp_path / "results.json"
    results = [
        _result("recursive_512", "bge-small", "dense", 0.8, 0.7, 30),
        _result("fixed_512", "bge-small", "bm25", 0.7, 0.6, 10),
    ]

    payload = generate_json_report(results, str(output_path))

    assert output_path.exists()
    assert payload["configuration_count"] == 2
    assert payload["recommendation"]["config_id"] == "recursive_512|bge-small|dense|none"
    assert payload["recommendation"]["latency_delta_ms_vs_fastest"] == 20
    assert payload["results"][0]["rank"] == 1
    assert payload["results"][0]["context_precision"] == 0.5
