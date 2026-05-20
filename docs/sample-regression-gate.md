# RAG Forge Regression Gate

**Verdict:** `pass`

## Recommendation

| Field | Baseline | Current |
|---|---:|---:|
| Config | `recursive_512|bge-small|dense|none` | `recursive_512|bge-small|dense|none` |
| Hit rate | 0.65 | 0.65 |
| MRR | 0.6 | 0.6 |
| Latency | 10.7ms | 10.7ms |
| Configurations | 24 | 24 |

## Checks

| Check | Status | Observed | Threshold |
|---|---|---:|---:|
| `hit_rate_drop` | `pass` | 0 | 0.02 |
| `mrr_drop` | `pass` | 0 | 0.02 |
| `latency_increase_pct` | `pass` | 0 | 25 |
| `recommendation_changed` | `pass` | false | `same recommended config` |
| `configuration_count_changed` | `pass` | 0 | 0 |
