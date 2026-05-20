# Verification

Last local verification: 2026-05-20

## Unit and Lint Gate

Command:

```bash
PYTHON=.venv/bin/python make verify
```

Result:

- Tests: 37 passed
- Ruff: clean

## Keyless Sample Benchmark

Command:

```bash
rm -rf /tmp/rag-forge-sample-smoke-check
PYTHON=.venv/bin/python ./scripts/run-sample-benchmark.sh /tmp/rag-forge-sample-smoke-check
```

Scope:

- Documents: `data/sample` (`3` files)
- QA pairs: `data/sample/qa.csv` (`20` questions)
- Configurations: `24`
- OpenAI embeddings: skipped
- Cross-encoder reranker: skipped

Result:

- Best configuration: `semantic|e5-small|hybrid|none`
- Best hit rate: `0.650`
- Best MRR: `0.617`
- Cached recommendation query latency on this run: `13.0ms`
- Artifacts generated: `results.md`, `results.json`, `pareto.png`

## Regression Gate

Command:

```bash
PYTHON=.venv/bin/python -m rag_forge.cli gate \
  --baseline /tmp/rag-forge-sample-smoke-check/results.json \
  --current /tmp/rag-forge-sample-smoke-check/results.json \
  --output docs/sample-regression-gate.json \
  --markdown docs/sample-regression-gate.md
```

Result:

- Verdict: `pass`
- Maximum allowed hit-rate drop: `0.02`
- Maximum allowed MRR drop: `0.02`
- Maximum allowed latency increase: `25%`

This is a local proof artifact. The sample corpus is intentionally small, and latency
excludes corpus loading, chunking, corpus embedding, first model download, and report
rendering. The numbers should be read as smoke-test evidence for the runner and gate
rather than a general retrieval benchmark.
