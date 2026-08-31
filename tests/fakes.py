"""Shared test doubles."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np


class FakeModel:
    """Records what it was asked to encode; returns one fake vector per text.

    Returns a `numpy.float32` array, like the real `SentenceTransformer.encode`
    does, so tests that assert on the returned element type (native `float`,
    not `numpy.float32`) actually exercise that conversion.
    """

    def __init__(self) -> None:
        self.encode_calls: list[list[str]] = []

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> Any:
        self.encode_calls.append(list(texts))
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


class FakeChatModel:
    """Records the prompts it was asked to answer; returns a canned reply."""

    def __init__(self, answer: str = "Réponse générée.") -> None:
        self.answer = answer
        self.invoke_calls: list[str] = []

    def invoke(self, prompt: str) -> Any:
        self.invoke_calls.append(prompt)
        return SimpleNamespace(content=self.answer)
