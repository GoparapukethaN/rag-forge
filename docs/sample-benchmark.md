# Sample Benchmark Smoke Test

This is a small local smoke test, not a universal RAG benchmark. It exists to prove the
benchmark runner can execute end to end without paid API keys.

## Command

```bash
PYTHON=.venv/bin/python ./scripts/run-sample-benchmark.sh /tmp/rag-forge-sample-smoke
```

## Latest Local Run

Date: 2026-05-20

Input:

- Documents: `data/sample` (`3` files)
- QA pairs: `data/sample/qa.csv` (`20` questions)
- API keys: none required
- OpenAI embeddings: skipped
- Cross-encoder reranker: skipped

Result:

- Configurations tested: `24`
- Best hit rate: `0.650`
- Best MRR: `0.617`
- Report artifacts: `results.md`, `results.json`, `pareto.png`
- Regression gate artifact: [sample-regression-gate.md](sample-regression-gate.md)

Recommendation from the generated report:

```text
Use semantic|e5-small|hybrid|none as the current default candidate.
It has the highest hit rate (0.650) and MRR (0.617) in this run.
```

Top configurations from the latest local run:

| Rank | Chunker | Embedder | Retriever | Reranker | Hit Rate | MRR | Cached Query Latency |
|---:|---|---|---|---|---:|---:|---:|
| 1 | semantic | e5-small | hybrid | none | 0.650 | 0.617 | 13ms |
| 2 | fixed_512 | e5-small | dense | none | 0.650 | 0.600 | 70ms |
| 3 | recursive_512 | bge-small | dense | none | 0.650 | 0.600 | 14ms |
| 4 | recursive_512 | e5-small | dense | none | 0.650 | 0.600 | 14ms |
| 5 | recursive_512 | e5-small | hybrid | none | 0.650 | 0.600 | 11ms |

Latency is cached query timing from a local smoke run. It excludes corpus loading,
chunking, corpus embedding, first model download, and report rendering, so it should be
treated as directional.

The JSON report is intended for follow-up analysis or dashboards. Its top-level shape is:

```json
{
  "configuration_count": 24,
  "recommendation": {
    "config_id": "semantic|e5-small|hybrid|none",
    "reason": "highest_hit_rate_then_mrr",
    "hit_rate": 0.65,
    "mrr": 0.6167,
    "context_precision": 0.13
  },
  "results": []
}
```

## Regression Gate

The same report can be used for a self-comparison smoke check. In normal use, compare
the latest accepted `results.json` against a new run:

```bash
rag-forge gate \
  --baseline /tmp/rag-forge-sample-smoke/results.json \
  --current /tmp/rag-forge-sample-smoke/results.json \
  --output docs/sample-regression-gate.json \
  --markdown docs/sample-regression-gate.md
```

The gate fails on quality regressions beyond the configured hit-rate or MRR thresholds,
fails on large latency increases, and warns when the winning retrieval configuration or
benchmark grid changes.
