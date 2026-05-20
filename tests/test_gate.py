import json
from pathlib import Path

from typer.testing import CliRunner

from rag_forge.bench import RunResult
from rag_forge.cli import app
from rag_forge.evaluate import EvalResult
from rag_forge.gate import (
    GateThresholds,
    evaluate_regression_gate,
    generate_gate_markdown,
)
from rag_forge.report import generate_json_report


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


def _report(hit_rate: float, mrr: float, latency_ms: float) -> dict:
    return generate_json_report([
        _result("recursive_512", "bge-small", "dense", hit_rate, mrr, latency_ms),
        _result("fixed_512", "bge-small", "bm25", hit_rate - 0.1, mrr - 0.1, 8),
    ])


def test_regression_gate_passes_when_current_is_within_thresholds() -> None:
    baseline = _report(hit_rate=0.8, mrr=0.7, latency_ms=20)
    current = _report(hit_rate=0.79, mrr=0.69, latency_ms=24)

    gate = evaluate_regression_gate(
        baseline,
        current,
        thresholds=GateThresholds(
            max_hit_rate_drop=0.02,
            max_mrr_drop=0.02,
            max_latency_increase_pct=25,
        ),
    )

    assert gate["verdict"] == "pass"
    assert {check["name"]: check["status"] for check in gate["checks"]} == {
        "hit_rate_drop": "pass",
        "mrr_drop": "pass",
        "latency_increase_pct": "pass",
        "recommendation_changed": "pass",
        "configuration_count_changed": "pass",
    }


def test_regression_gate_fails_when_quality_or_latency_regresses() -> None:
    baseline = _report(hit_rate=0.8, mrr=0.7, latency_ms=20)
    current = _report(hit_rate=0.72, mrr=0.61, latency_ms=34)

    gate = evaluate_regression_gate(
        baseline,
        current,
        thresholds=GateThresholds(
            max_hit_rate_drop=0.02,
            max_mrr_drop=0.02,
            max_latency_increase_pct=25,
        ),
    )

    checks = {check["name"]: check for check in gate["checks"]}
    assert gate["verdict"] == "fail"
    assert checks["hit_rate_drop"]["status"] == "fail"
    assert checks["mrr_drop"]["status"] == "fail"
    assert checks["latency_increase_pct"]["status"] == "fail"
    assert checks["hit_rate_drop"]["observed"] == 0.08
    assert checks["latency_increase_pct"]["observed"] == 70.0


def test_regression_gate_warns_when_winner_changes_without_metric_drop() -> None:
    baseline = generate_json_report([
        _result("recursive_512", "bge-small", "dense", 0.8, 0.7, 20),
        _result("fixed_512", "bge-small", "bm25", 0.7, 0.6, 10),
    ])
    current = generate_json_report([
        _result("semantic", "bge-small", "hybrid", 0.81, 0.71, 19),
        _result("recursive_512", "bge-small", "dense", 0.8, 0.7, 20),
    ])

    gate = evaluate_regression_gate(baseline, current)

    checks = {check["name"]: check for check in gate["checks"]}
    assert gate["verdict"] == "warn"
    assert checks["recommendation_changed"]["status"] == "warn"
    assert checks["recommendation_changed"]["baseline"] == "recursive_512|bge-small|dense|none"
    assert checks["recommendation_changed"]["current"] == "semantic|bge-small|hybrid|none"


def test_markdown_gate_report_summarizes_verdict_and_checks() -> None:
    gate = evaluate_regression_gate(
        _report(hit_rate=0.8, mrr=0.7, latency_ms=20),
        _report(hit_rate=0.72, mrr=0.61, latency_ms=34),
    )

    markdown = generate_gate_markdown(gate)

    assert "# RAG Forge Regression Gate" in markdown
    assert "**Verdict:** `fail`" in markdown
    assert "| Check | Status | Observed | Threshold |" in markdown
    assert "`hit_rate_drop`" in markdown


def test_gate_cli_writes_reports_and_exits_nonzero_on_failure(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    output_path = tmp_path / "gate.json"
    markdown_path = tmp_path / "gate.md"
    baseline_path.write_text(json.dumps(_report(0.8, 0.7, 20)), encoding="utf-8")
    current_path.write_text(json.dumps(_report(0.72, 0.61, 34)), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "gate",
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--output",
            str(output_path),
            "--markdown",
            str(markdown_path),
        ],
    )

    assert result.exit_code == 1
    assert "Regression gate failed" in result.output
    assert json.loads(output_path.read_text(encoding="utf-8"))["verdict"] == "fail"
    assert "# RAG Forge Regression Gate" in markdown_path.read_text(encoding="utf-8")
