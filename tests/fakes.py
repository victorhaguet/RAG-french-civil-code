"""Shared test doubles."""

from __future__ import annotations

import re
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np


class FakeModel:
    """Records each `encode` call (texts and normalization); returns one fake vector per text.

    Returns a `numpy.float32` array, like the real `SentenceTransformer.encode`
    does, so tests that assert on the returned element type (native `float`,
    not `numpy.float32`) actually exercise that conversion.
    """

    def __init__(self) -> None:
        self.encode_calls: list[list[str]] = []
        self.normalize_embeddings_calls: list[bool] = []

    # Defaults to False, like the real `SentenceTransformer.encode`, so a
    # caller that forgets to ask for normalized embeddings is recorded as such.
    def encode(self, texts: list[str], normalize_embeddings: bool = False) -> Any:
        self.encode_calls.append(list(texts))
        self.normalize_embeddings_calls.append(normalize_embeddings)
        return np.array([[float(len(text))] for text in texts], dtype=np.float32)


class FakeCrossEncoder:
    """Records the pairs it was asked to score; returns one fake score per pair.

    The score is derived from the input (`(query, document)` text length),
    like `FakeModel` derives its fake vectors from text length, so tests can
    assert on a specific reranked order without hardcoding opaque numbers.
    """

    def __init__(self) -> None:
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> Any:
        self.predict_calls.append(list(pairs))
        return np.array(
            [float(len(query) + len(document)) for query, document in pairs],
            dtype=np.float32,
        )


class PassthroughCrossEncoder:
    """Scores pairs by descending input order, so reranking is a no-op.

    Used as the default Reranker in tests that predate reranking and assert on
    fusion order directly.
    """

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [-index for index in range(len(pairs))]


class FakeChatModel:
    """Records the prompts it was asked to answer; returns a canned reply.

    `stream` yields the same reply in several pieces (one per word, trailing
    whitespace included), so the pieces joined are the reply. With
    `fail_after`, it raises after yielding that many pieces, like a model
    failing mid-generation.
    """

    FAILURE_DETAIL = "fake chat model failure: internal detail"

    def __init__(self, answer: str = "Réponse générée.", fail_after: int | None = None) -> None:
        self.answer = answer
        self.fail_after = fail_after
        self.invoke_calls: list[str] = []
        self.stream_calls: list[str] = []

    def invoke(self, prompt: str) -> Any:
        self.invoke_calls.append(prompt)
        return SimpleNamespace(content=self.answer)

    def stream(self, prompt: str) -> Iterator[Any]:
        self.stream_calls.append(prompt)
        for index, piece in enumerate(re.findall(r"\S+\s*", self.answer)):
            if index == self.fail_after:
                raise RuntimeError(self.FAILURE_DETAIL)
            yield SimpleNamespace(content=piece)
