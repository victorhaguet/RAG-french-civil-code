"""Shared test doubles."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class FakeModel:
    """Records what it was asked to encode; returns one fake vector per text."""

    def __init__(self) -> None:
        self.encode_calls: list[list[str]] = []

    def encode(
        self, texts: list[str], normalize_embeddings: bool = True
    ) -> list[list[float]]:
        self.encode_calls.append(list(texts))
        return [[float(len(text))] for text in texts]


class FakeChatModel:
    """Records the prompts it was asked to answer; returns a canned reply."""

    def __init__(self, answer: str = "Réponse générée.") -> None:
        self.answer = answer
        self.invoke_calls: list[str] = []

    def invoke(self, prompt: str) -> Any:
        self.invoke_calls.append(prompt)
        return SimpleNamespace(content=self.answer)
