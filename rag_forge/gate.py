"""Regression gate for comparing benchmark reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateThresholds:
    max_hit_rate_drop: float = 0.02
    max_mrr_drop: float = 0.02
    max_latency_increase_pct: float = 25.0


def evaluate_regression_gate(
    baseline: dict[str, Any],
    current: dict[str, Any],
    thresholds: GateThresholds = GateThresholds(),
) -> dict[str, Any]:
    baseline_rec = baseline.get("recommendation") or {}
    current_rec = current.get("recommendation") or {}
    checks = [
        _quality_check(
            "hit_rate_drop",
            baseline_rec.get("hit_rate"),
            current_rec.get("hit_rate"),
            thresholds.max_hit_rate_drop,
        ),
        _quality_check(
            "mrr_drop",
            baseline_rec.get("mrr"),
            current_rec.get("mrr"),
            thresholds.max_mrr_drop,
        ),
        _latency_check(
            baseline_rec.get("latency_ms"),
            current_rec.get("latency_ms"),
            thresholds.max_latency_increase_pct,
        ),
        _recommendation_check(
            baseline_rec.get("config_id"),
            current_rec.get("config_id"),
        ),
        _configuration_count_check(
            baseline.get("configuration_count"),
            current.get("configuration_count"),
        ),
    ]
    verdict = _verdict(checks)
    return {
        "verdict": verdict,
        "thresholds": {
            "max_hit_rate_drop": thresholds.max_hit_rate_drop,
            "max_mrr_drop": thresholds.max_mrr_drop,
            "max_latency_increase_pct": thresholds.max_latency_increase_pct,
        },
        "baseline": _summary(baseline),
        "current": _summary(current),
        "checks": checks,
    }


def generate_gate_markdown(gate: dict[str, Any]) -> str:
    baseline = gate["baseline"]
    current = gate["current"]
    lines = [
        "# RAG Forge Regression Gate",
        "",
        f"**Verdict:** `{gate['verdict']}`",
        "",
        "## Recommendation",
        "",
        "| Field | Baseline | Current |",
        "|---|---:|---:|",
        f"| Config | `{baseline['config_id']}` | `{current['config_id']}` |",
        f"| Hit rate | {_format_number(baseline['hit_rate'])} | "
        f"{_format_number(current['hit_rate'])} |",
        f"| MRR | {_format_number(baseline['mrr'])} | {_format_number(current['mrr'])} |",
        f"| Cached query latency | {_format_number(baseline['latency_ms'])}ms | "
        f"{_format_number(current['latency_ms'])}ms |",
        f"| Configurations | {baseline['configuration_count']} | "
        f"{current['configuration_count']} |",
        "",
        "## Checks",
        "",
        "| Check | Status | Observed | Threshold |",
        "|---|---|---:|---:|",
    ]
    for check in gate["checks"]:
        lines.append(
            f"| `{check['name']}` | `{check['status']}` | "
            f"{_format_value(check.get('observed'))} | {_format_value(check.get('threshold'))} |"
        )
    return "\n".join(lines) + "\n"


def load_json_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_gate_json(gate: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")


def write_gate_markdown(gate: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_gate_markdown(gate), encoding="utf-8")


def _quality_check(
    name: str,
    baseline_value: float | None,
    current_value: float | None,
    threshold: float,
) -> dict[str, Any]:
    if baseline_value is None or current_value is None:
        return {
            "name": name,
            "status": "fail",
            "observed": None,
            "threshold": threshold,
            "baseline": baseline_value,
            "current": current_value,
        }
    observed = round(float(baseline_value) - float(current_value), 4)
    return {
        "name": name,
        "status": "pass" if observed <= threshold else "fail",
        "observed": observed,
        "threshold": threshold,
        "baseline": baseline_value,
        "current": current_value,
    }


def _latency_check(
    baseline_value: float | None,
    current_value: float | None,
    threshold: float,
) -> dict[str, Any]:
    if baseline_value is None or current_value is None:
        observed = None
        status = "fail"
    else:
        baseline_latency = float(baseline_value)
        current_latency = float(current_value)
        if baseline_latency <= 0:
            observed = 0.0 if current_latency <= baseline_latency else 100.0
        else:
            observed = round(((current_latency - baseline_latency) / baseline_latency) * 100, 1)
        status = "pass" if observed <= threshold else "fail"
    return {
        "name": "latency_increase_pct",
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "baseline": baseline_value,
        "current": current_value,
    }


def _recommendation_check(
    baseline_config_id: str | None,
    current_config_id: str | None,
) -> dict[str, Any]:
    status = "pass" if baseline_config_id == current_config_id else "warn"
    if not baseline_config_id or not current_config_id:
        status = "fail"
    return {
        "name": "recommendation_changed",
        "status": status,
        "observed": baseline_config_id != current_config_id,
        "threshold": "same recommended config",
        "baseline": baseline_config_id,
        "current": current_config_id,
    }


def _configuration_count_check(
    baseline_count: int | None,
    current_count: int | None,
) -> dict[str, Any]:
    status = "pass" if baseline_count == current_count else "warn"
    if baseline_count is None or current_count is None:
        status = "fail"
    observed = (
        None if baseline_count is None or current_count is None else current_count - baseline_count
    )
    return {
        "name": "configuration_count_changed",
        "status": status,
        "observed": observed,
        "threshold": 0,
        "baseline": baseline_count,
        "current": current_count,
    }


def _verdict(checks: list[dict[str, Any]]) -> str:
    statuses = {check["status"] for check in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    recommendation = report.get("recommendation") or {}
    return {
        "configuration_count": report.get("configuration_count"),
        "config_id": recommendation.get("config_id"),
        "hit_rate": recommendation.get("hit_rate"),
        "mrr": recommendation.get("mrr"),
        "context_precision": recommendation.get("context_precision"),
        "latency_ms": recommendation.get("latency_ms"),
    }


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"`{value}`"
