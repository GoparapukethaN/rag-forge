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
- Best MRR: `0.600`
- Report artifacts: `results.md`, `results.json`, `pareto.png`
- Regression gate artifact: [sample-regression-gate.md](sample-regression-gate.md)

Recommendation from the generated report:

```text
Use recursive_512|bge-small|dense|none as the current default candidate.
It has the highest hit rate (0.650) and MRR (0.600) in this run.
```

Top configurations from the latest local run:

| Rank | Chunker | Embedder | Retriever | Reranker | Hit Rate | MRR | Latency |
|---:|---|---|---|---|---:|---:|---:|
| 1 | recursive_512 | bge-small | dense | none | 0.650 | 0.600 | 14ms |
| 2 | recursive_512 | e5-small | dense | none | 0.650 | 0.600 | 14ms |
| 3 | recursive_512 | bge-small | hybrid | none | 0.650 | 0.592 | 14ms |
| 4 | recursive_512 | e5-small | hybrid | none | 0.650 | 0.562 | 13ms |
| 5 | fixed_512 | bge-small | hybrid | none | 0.650 | 0.560 | 12ms |

Latency is local-machine timing from a smoke run and should be treated as directional.

The JSON report is intended for follow-up analysis or dashboards. Its top-level shape is:

```json
{
  "configuration_count": 24,
  "recommendation": {
    "config_id": "recursive_512|bge-small|dense|none",
    "reason": "highest_hit_rate_then_mrr",
    "hit_rate": 0.65,
    "mrr": 0.6,
    "context_precision": 0.13
  },
  "results": []
}
```

## Regression Gate

The same report can be used as a baseline for future benchmark runs:

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
