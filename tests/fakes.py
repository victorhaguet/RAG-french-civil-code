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


class FakeChatModel:
    """Records the prompts it was asked to answer; returns a canned reply."""

    def __init__(self, answer: str = "Réponse générée.") -> None:
        self.answer = answer
        self.invoke_calls: list[str] = []

    def invoke(self, prompt: str) -> Any:
        self.invoke_calls.append(prompt)
        return SimpleNamespace(content=self.answer)
