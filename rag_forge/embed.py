"""Embedding functions.

BGE-small and E5-small run locally after the first model download. OpenAI needs
OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# lazy-loaded models to avoid import-time downloads
_models: dict = {}


def _get_st_model(model_name: str):
    """Load a sentence-transformers model, cached."""
    if model_name not in _models:
        from sentence_transformers import SentenceTransformer

        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def bge_embed(texts: list[str]) -> np.ndarray:
    """Embed with BAAI/bge-small-en-v1.5 (local, 384-dim).

    The small variant keeps the local benchmark easier to run.
    """
    model = _get_st_model("BAAI/bge-small-en-v1.5")
    # BGE models want "Represent this sentence:" prefix for retrieval
    prefixed = [f"Represent this sentence: {t}" for t in texts]
    embeddings = model.encode(prefixed, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings)


def e5_embed(texts: list[str]) -> np.ndarray:
    """Embed with intfloat/e5-small-v2 (local, 384-dim).

    E5 wants "query: " or "passage: " prefixes. We use "passage: " for indexing
    and "query: " gets applied at search time in retrieve.py.
    """
    model = _get_st_model("intfloat/e5-small-v2")
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(prefixed, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings)


def openai_embed(texts: list[str]) -> np.ndarray:
    """Embed with OpenAI text-embedding-3-small (API, 1536-dim).

    Best quality in our benchmarks but costs money and adds latency.
    Set OPENAI_API_KEY env var to use this.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Either set it or use bge/e5 embedders instead."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    # batch in groups of 100 to avoid API limits
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)


# name -> (embed_fn, dimension, needs_query_prefix)
EMBEDDERS = {
    "bge-small": (bge_embed, 384, False),
    "e5-small": (e5_embed, 384, True),  # needs "query: " prefix at search time
    "openai": (openai_embed, 1536, False),
}
