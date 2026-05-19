#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${1:-results/sample-smoke}"
MPLCONFIGDIR="$(mktemp -d "${TMPDIR:-/tmp}/rag-forge-mpl.XXXXXX")"
export MPLCONFIGDIR

"$PYTHON" -m rag_forge.cli run \
  --docs ./data/sample \
  --qa ./data/sample/qa.csv \
  --output "$OUTPUT_DIR" \
  --skip-openai \
  --skip-reranker
