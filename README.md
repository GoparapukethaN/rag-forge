# RAG Forge

RAG Forge is a small benchmark runner for comparing retrieval pipeline choices:
chunking strategy, embedding model, retrieval method, and optional reranking.

The goal is simple: make RAG configuration changes measurable instead of guessing from a
few manual questions.

## What It Does

Give RAG Forge a directory of `.txt` or `.md` documents and a CSV of question/answer
pairs. It builds a retrieval benchmark across combinations of:

- **Chunking:** fixed-size, recursive, and semantic-style paragraph grouping
- **Embeddings:** local BGE-small, local E5-small, and optional OpenAI embeddings
- **Retrieval:** dense, BM25, and hybrid retrieval
- **Reranking:** cross-encoder reranking or no reranker

For each configuration it records hit rate, MRR, context precision, and average retrieval
latency, then writes a Markdown report and optional Pareto plot.

## Quick Start

```bash
git clone https://github.com/GoparapukethaN/rag-forge.git
cd rag-forge

python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"

rag-forge run --docs ./data/sample --qa ./data/sample/qa.csv --skip-openai
```

The local embedding models download on first use. Use `--skip-openai` to keep the run
keyless.

## CLI Reference

```bash
# run the included sample benchmark
rag-forge run --docs ./data/sample --qa ./data/sample/qa.csv --skip-openai

# skip reranking for a faster run
rag-forge run --docs ./data/sample --qa ./data/sample/qa.csv \
  --skip-openai \
  --skip-reranker

# custom output directory and retrieval depth
rag-forge run --docs ./my_docs --qa ./my_qa.csv --output ./my_results --top-k 10
```

## QA File Format

The CSV needs `question` and `answer` columns:

```csv
question,answer
What is RAG?,Retrieval-Augmented Generation combines retrieval with generation
What metric checks ranking position?,MRR
```

The evaluation checks whether the retrieved chunks contain the expected answer text. This
makes the benchmark retrieval-focused; it does not score generated responses.

## Verification

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
make verify
```

Last local verification (2026-05-20): `28 passed` and `ruff` clean.

For the included keyless sample benchmark:

```bash
PYTHON=.venv/bin/python ./scripts/run-sample-benchmark.sh /tmp/rag-forge-sample-smoke
```

Sample smoke result from 2026-05-20: 24 configurations tested, best hit rate `0.650`,
best MRR `0.600`. See [docs/sample-benchmark.md](docs/sample-benchmark.md) for the
exact command, scope, and top configurations.

## How It Works

1. Load documents and QA pairs.
2. Chunk each document with every configured chunking strategy.
3. Embed chunks for each configured embedder.
4. Run dense, sparse, or hybrid retrieval for every question.
5. Optionally rerank the retrieved chunks.
6. Score retrieval against the expected answer text.
7. Rank configurations and generate a report.

Embedding work is cached within a benchmark run so retrieval methods and rerankers can
reuse the same chunk embeddings.

## Limitations

- Only `.txt` and `.md` files are supported.
- Local embedding models require a first-run model download.
- Evaluation is retrieval-only; generation quality is out of scope for this version.
- The sample dataset is intentionally small and should be treated as a smoke test, not a
  universal benchmark.
- Designed for English text.

## License

MIT. See [LICENSE](LICENSE).
