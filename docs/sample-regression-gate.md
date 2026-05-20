# RAG Forge Regression Gate

**Verdict:** `pass`

## Recommendation

| Field | Baseline | Current |
|---|---:|---:|
| Config | `semantic|e5-small|hybrid|none` | `semantic|e5-small|hybrid|none` |
| Hit rate | 0.65 | 0.65 |
| MRR | 0.617 | 0.617 |
| Cached query latency | 13ms | 13ms |
| Configurations | 24 | 24 |

## Checks

| Check | Status | Observed | Threshold |
|---|---|---:|---:|
| `hit_rate_drop` | `pass` | 0 | 0.02 |
| `mrr_drop` | `pass` | 0 | 0.02 |
| `latency_increase_pct` | `pass` | 0 | 25 |
| `recommendation_changed` | `pass` | false | `same recommended config` |
| `configuration_count_changed` | `pass` | 0 | 0 |
