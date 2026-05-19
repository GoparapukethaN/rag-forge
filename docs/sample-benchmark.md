# Sample Benchmark Smoke Test

This is a small local smoke test, not a universal RAG benchmark. It exists to prove the
benchmark runner can execute end to end without paid API keys.

## Command

```bash
PYTHON=.venv/bin/python ./scripts/run-sample-benchmark.sh /tmp/rag-forge-sample-smoke
```

## Latest Local Run

Date: 2026-05-19

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

Top configurations from the latest local run:

| Rank | Chunker | Embedder | Retriever | Reranker | Hit Rate | MRR | Latency |
|---:|---|---|---|---|---:|---:|---:|
| 1 | recursive_512 | bge-small | dense | none | 0.650 | 0.600 | 11ms |
| 2 | recursive_512 | e5-small | dense | none | 0.650 | 0.600 | 11ms |
| 3 | recursive_512 | bge-small | hybrid | none | 0.650 | 0.592 | 11ms |
| 4 | recursive_512 | e5-small | hybrid | none | 0.650 | 0.562 | 11ms |
| 5 | fixed_512 | bge-small | hybrid | none | 0.650 | 0.560 | 12ms |

Latency is local-machine timing from a smoke run and should be treated as directional.
