"""The benchmark runner. This is the whole point of rag-forge.

Takes documents + QA pairs, runs every combination of chunking × embedding ×
retrieval × reranking, evaluates each combo, and returns ranked results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

from rag_forge.chunk import CHUNKERS
from rag_forge.embed import EMBEDDERS
from rag_forge.evaluate import EvalResult, evaluate_retrieval
from rag_forge.rerank import RERANKERS
from rag_forge.retrieve import bm25_search, dense_search, hybrid_search


@dataclass
class BenchConfig:
    chunkers: list[str] = field(default_factory=lambda: list(CHUNKERS.keys()))
    embedders: list[str] = field(default_factory=lambda: list(EMBEDDERS.keys()))
    retrievers: list[str] = field(default_factory=lambda: ["dense", "bm25", "hybrid"])
    rerankers: list[str] = field(default_factory=lambda: list(RERANKERS.keys()))
    top_k: int = 5


@dataclass
class RunResult:
    chunker: str
    embedder: str
    retriever: str
    reranker: str
    eval: EvalResult
    latency_ms: float  # avg retrieval latency per query
    num_chunks: int
    config_id: str = ""

    def __post_init__(self):
        self.config_id = f"{self.chunker}|{self.embedder}|{self.retriever}|{self.reranker}"


def load_documents(docs_path: str | Path) -> list[str]:
    """Load all .txt and .md files from a directory. Nothing fancy."""
    docs_path = Path(docs_path)
    documents = []
    for ext in ["*.txt", "*.md"]:
        for f in sorted(docs_path.glob(ext)):
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                documents.append(text)
    if not documents:
        raise FileNotFoundError(f"No .txt or .md files found in {docs_path}")
    return documents


def load_qa_pairs(qa_path: str | Path) -> tuple[list[str], list[str]]:
    """Load question-answer pairs from CSV. Expects columns: question,answer"""
    import csv

    qa_path = Path(qa_path)
    questions, answers = [], []
    with open(qa_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            a = row.get("answer", "").strip()
            if q and a:
                questions.append(q)
                answers.append(a)
    if not questions:
        raise ValueError(
            f"No valid QA pairs in {qa_path}. Need 'question','answer' columns."
        )
    return questions, answers


def run_benchmark(
    documents: list[str],
    questions: list[str],
    ground_truths: list[str],
    config: BenchConfig | None = None,
    verbose: bool = True,
) -> list[RunResult]:
    """Run the full benchmark grid and return ranked results.

    This is the main function. It:
    1. Chunks documents with each chunking strategy
    2. Embeds chunks with each embedding model
    3. Retrieves for each query with each retrieval method
    4. Optionally reranks
    5. Evaluates each combination
    """
    if config is None:
        config = BenchConfig()

    # filter out openai embedder if no key
    import os

    if "openai" in config.embedders and not os.environ.get("OPENAI_API_KEY"):
        config.embedders = [e for e in config.embedders if e != "openai"]
        if verbose:
            print("  [skip] openai embedder (no OPENAI_API_KEY)")

    total = (
        len(config.chunkers)
        * len(config.embedders)
        * len(config.retrievers)
        * len(config.rerankers)
    )
    if verbose:
        print(f"\nRunning {total} configurations...")
        print(f"  {len(config.chunkers)} chunkers × {len(config.embedders)} embedders "
              f"× {len(config.retrievers)} retrievers × {len(config.rerankers)} rerankers\n")

    results = []
    run_num = 0

    # precompute chunks for each chunker (reused across embedders)
    chunk_cache: dict[str, list[str]] = {}
    for chunker_name in config.chunkers:
        chunker_fn = CHUNKERS[chunker_name]
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunker_fn(doc))
        chunk_cache[chunker_name] = all_chunks
        if verbose:
            print(f"  chunked with {chunker_name}: {len(all_chunks)} chunks")

    # precompute embeddings for each (chunker, embedder) pair
    embedding_cache: dict[str, np.ndarray] = {}
    for chunker_name in config.chunkers:
        chunks = chunk_cache[chunker_name]
        for embedder_name in config.embedders:
            cache_key = f"{chunker_name}|{embedder_name}"
            embed_fn, _, _ = EMBEDDERS[embedder_name]
            if verbose:
                print(f"  embedding {len(chunks)} chunks with {embedder_name}...",
                      end=" ", flush=True)
            t0 = time.time()
            embedding_cache[cache_key] = embed_fn(chunks)
            dt = time.time() - t0
            if verbose:
                print(f"({dt:.1f}s)")

    # now run every combination
    combos = list(product(config.chunkers, config.embedders, config.retrievers, config.rerankers))
    for chunker_name, embedder_name, retriever_name, reranker_name in combos:
        run_num += 1
        chunks = chunk_cache[chunker_name]
        embed_key = f"{chunker_name}|{embedder_name}"
        corpus_embeddings = embedding_cache[embed_key]
        _, query_embed_fn, _ = EMBEDDERS[embedder_name]
        rerank_fn = RERANKERS[reranker_name]

        if verbose:
            print(f"  [{run_num}/{total}] {chunker_name} + {embedder_name} + "
                  f"{retriever_name} + {reranker_name}", end="")

        all_retrieved_contexts = []
        latencies = []

        for query in questions:
            t0 = time.time()

            # embed the query with the embedder-specific query path
            query_emb = query_embed_fn([query])[0]

            # retrieve
            if retriever_name == "dense":
                results_raw = dense_search(query_emb, corpus_embeddings, top_k=config.top_k * 2)
            elif retriever_name == "bm25":
                results_raw = bm25_search(query, chunks, top_k=config.top_k * 2)
            elif retriever_name == "hybrid":
                results_raw = hybrid_search(
                    query, query_emb, chunks, corpus_embeddings, top_k=config.top_k * 2
                )
            else:
                raise ValueError(f"Unknown retriever: {retriever_name}")

            # rerank
            retrieved_indices = [idx for idx, _ in results_raw]
            retrieved_chunks = [chunks[idx] for idx in retrieved_indices]
            reranked = rerank_fn(query, retrieved_chunks, retrieved_indices, top_k=config.top_k)

            final_chunks = [chunks[idx] for idx, _ in reranked]
            all_retrieved_contexts.append(final_chunks)

            latencies.append((time.time() - t0) * 1000)

        # evaluate
        eval_result = evaluate_retrieval(
            questions, ground_truths, all_retrieved_contexts, config.top_k
        )
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        result = RunResult(
            chunker=chunker_name,
            embedder=embedder_name,
            retriever=retriever_name,
            reranker=reranker_name,
            eval=eval_result,
            latency_ms=round(avg_latency, 1),
            num_chunks=len(chunks),
        )
        results.append(result)

        if verbose:
            print(f" → hit_rate={eval_result.hit_rate:.2f}, mrr={eval_result.mrr:.2f}, "
                  f"latency={avg_latency:.0f}ms")

    # sort by hit_rate desc, then mrr desc
    results.sort(key=lambda r: (r.eval.hit_rate, r.eval.mrr), reverse=True)
    return results
