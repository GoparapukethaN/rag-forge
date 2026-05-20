# RAG Forge Case Study

RAG Forge is my local benchmark harness for retrieval changes. The project came from a
simple problem I kept running into: RAG demos can feel good after a few manual questions,
but it is hard to tell whether a chunking, embedding, or retrieval change made the system
better.

This project keeps the scope narrow on purpose. It measures retrieval quality, writes
reviewable reports, and fails a regression gate when a new run drops below the accepted
baseline.

## Problem

A retrieval pipeline has several choices that interact with each other:

- fixed, recursive, or paragraph-aware chunking
- local BGE/E5 embeddings or optional hosted embeddings
- dense, BM25, or hybrid retrieval
- reranking or no reranking

Testing one question at a time does not show whether the change helped the corpus or only
one happy-path prompt. I wanted a small tool that could answer: "Did this retrieval
configuration improve the benchmark, and can I prove that with an artifact?"

## What I Built

The CLI runs a benchmark grid over a document directory and a CSV of question/answer
pairs. For every configuration it records:

- hit rate
- MRR
- context precision
- chunk count
- cached query latency

It writes both Markdown and JSON reports. The Markdown file is easy to review, while the
JSON file is meant for automation, dashboards, or future CI gates.

The second CLI path compares a baseline `results.json` against a current `results.json`.
The gate fails on hit-rate drops, MRR drops, large latency increases, recommendation
changes, or benchmark-grid changes.

## Current Sample Result

The included keyless sample uses `3` small documents and `20` QA pairs. The local smoke
run covers `24` retrieval configurations without an API key.

Latest local sample:

| Candidate | Hit Rate | MRR | Cached Query Latency |
| --- | ---: | ---: | ---: |
| `semantic|e5-small|hybrid|none` | 0.650 | 0.617 | 13ms |
| `fixed_512|e5-small|dense|none` | 0.650 | 0.600 | 70ms |
| `recursive_512|bge-small|dense|none` | 0.650 | 0.600 | 14ms |

The sample is intentionally small. I treat it as a smoke test for the harness, not a
universal retrieval benchmark.

## Design Choices

### Keep the default path keyless

The project can run locally with BGE/E5 embeddings and no hosted provider. Optional
OpenAI embeddings are isolated behind an extra install path and `OPENAI_API_KEY` check,
so the core project remains reproducible without credentials.

### Separate retrieval evaluation from answer generation

RAG Forge checks whether retrieved chunks contain expected evidence. It does not score a
generated answer. That boundary keeps the first version honest: it evaluates retrieval
quality before mixing in model style, model variance, or prompt phrasing.

### Store reports as artifacts

The Markdown report is for reading. The JSON report is for comparison. Keeping both makes
the project useful in two modes: an interview walkthrough and a regression gate.

### Make recommendation changes visible

The gate does more than compare scores. It also warns when the recommended configuration
changes or the benchmark grid changes, because those are release-review events even when
headline metrics still pass.

## Verification

Local verification currently includes:

- `37` pytest tests
- Ruff checks
- a keyless sample benchmark over `24` configurations
- a self-comparison regression gate artifact

Commands:

```bash
PYTHON=.venv/bin/python make verify
PYTHON=.venv/bin/python make sample-check
```

The repeatable proof is tracked in [verification.md](verification.md),
[sample-benchmark.md](sample-benchmark.md), and
[sample-regression-gate.md](sample-regression-gate.md).

## What I Would Improve Next

- Add a larger corpus with multiple document domains.
- Add optional answer-generation evaluation beside the retrieval-only score.
- Add a small dashboard for comparing benchmark runs over time.
- Add a reranker smoke path that is still practical on a normal laptop.
- Keep old accepted runs as named baselines instead of using ad hoc file paths.

