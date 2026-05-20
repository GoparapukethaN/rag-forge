import numpy as np

import rag_forge.bench as bench_module
from rag_forge import embed
from rag_forge.bench import BenchConfig, run_benchmark


class _FakeModel:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):  # noqa: ANN001
        self.inputs.append(list(texts))
        return np.ones((len(texts), 2))


def test_e5_uses_passage_and_query_prefixes(monkeypatch) -> None:
    model = _FakeModel()
    monkeypatch.setattr(embed, "_get_st_model", lambda _model_name: model)

    embed.e5_embed(["document text"])
    embed.e5_query_embed(["search question"])

    assert model.inputs == [["passage: document text"], ["query: search question"]]


def test_benchmark_uses_query_embedder_for_queries(monkeypatch) -> None:
    calls: dict[str, list[str]] = {"documents": [], "queries": []}

    def document_embed(texts: list[str]) -> np.ndarray:
        calls["documents"].extend(texts)
        return np.ones((len(texts), 2))

    def query_embed(texts: list[str]) -> np.ndarray:
        calls["queries"].extend(texts)
        return np.ones((len(texts), 2))

    monkeypatch.setitem(bench_module.EMBEDDERS, "fake-e5", (document_embed, query_embed, 2))

    run_benchmark(
        documents=["answer text"],
        questions=["search question"],
        ground_truths=["answer text"],
        config=BenchConfig(
            chunkers=["fixed_512"],
            embedders=["fake-e5"],
            retrievers=["dense"],
            rerankers=["none"],
            top_k=1,
        ),
        verbose=False,
    )

    assert calls["documents"] == ["answer text"]
    assert calls["queries"] == ["search question"]
