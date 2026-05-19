"""Reranking: cross-encoder or passthrough.

Cross-encoder reranking can improve retrieval quality, but the tradeoff is additional
latency per query.
"""

from __future__ import annotations

_reranker_model = None


def cross_encoder_rerank(
    query: str,
    chunks: list[str],
    chunk_indices: list[int],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Rerank retrieved chunks using a cross-encoder model.

    Uses ms-marco-MiniLM-L-6-v2 as a small local reranking baseline.
    """
    global _reranker_model

    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [[query, chunk] for chunk in chunks]
    scores = _reranker_model.predict(pairs)

    scored = list(zip(chunk_indices, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(idx, float(score)) for idx, score in scored[:top_k]]


def no_rerank(
    query: str,
    chunks: list[str],
    chunk_indices: list[int],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Passthrough — just return the chunks as-is with their original order."""
    return [(idx, 1.0 - i * 0.01) for i, idx in enumerate(chunk_indices[:top_k])]


RERANKERS = {
    "cross-encoder": cross_encoder_rerank,
    "none": no_rerank,
}
