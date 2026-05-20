#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${1:-$(mktemp -d "${TMPDIR:-/tmp}/rag-forge-sample.XXXXXX")}"
MPLCONFIGDIR="$(mktemp -d "${TMPDIR:-/tmp}/rag-forge-mpl.XXXXXX")"
export MPLCONFIGDIR

cleanup() {
  rm -rf "$MPLCONFIGDIR"
}
trap cleanup EXIT

"$PYTHON" -m rag_forge.cli run \
  --docs ./data/sample \
  --qa ./data/sample/qa.csv \
  --output "$OUTPUT_DIR" \
  --skip-openai \
  --skip-reranker

"$PYTHON" -m rag_forge.cli gate \
  --baseline "$OUTPUT_DIR/results.json" \
  --current "$OUTPUT_DIR/results.json" \
  --output "$OUTPUT_DIR/gate.json" \
  --markdown "$OUTPUT_DIR/gate.md"

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
results = json.loads((output_dir / "results.json").read_text())
gate = json.loads((output_dir / "gate.json").read_text())

assert results["configuration_count"] == 24, results["configuration_count"]
assert len(results["results"]) == 24, len(results["results"])
assert results["recommendation"]["config_id"], results["recommendation"]
assert (output_dir / "results.md").exists()
assert (output_dir / "pareto.png").exists()
assert gate["verdict"] == "pass", gate
assert all(check["status"] == "pass" for check in gate["checks"]), gate["checks"]
assert (output_dir / "gate.md").exists()
PY

echo "RAG Forge sample check passed in ${OUTPUT_DIR}"
