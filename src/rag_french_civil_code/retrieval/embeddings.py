"""Embeddings for multilingual-e5-large-instruct.

The model is asymmetric: document/passage text is embedded with no
instruction prefix, while queries need one (`"Instruct: {task}\\nQuery:
{query}"`) to retrieve well. Ticket 1 only needs the document side; the
query side here uses a fixed French instruction as a placeholder until the
query pipeline (Ticket 2) adapts it to the detected query language.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

DEFAULT_QUERY_INSTRUCTION = (
    "Étant donné une question juridique, retrouve les articles du Code "
    "civil pertinents pour y répondre."
)


class MultilingualE5Embeddings(Embeddings):
    """LangChain Embeddings backed by a local sentence-transformers model.

    `model` accepts anything duck-typing sentence-transformers' `.encode()`
    (a real `SentenceTransformer`, or a lightweight test double) — typed as
    `Any` since `SentenceTransformer.encode`'s real signature is a large
    overload set not worth mirroring here.
    """

    def __init__(self, model: Any | None = None) -> None:
        self._model: Any = model if model is not None else self._load_default_model()

    @staticmethod
    def _load_default_model() -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [list(vector) for vector in embeddings]

    def embed_query(self, text: str) -> list[float]:
        instructed = f"Instruct: {DEFAULT_QUERY_INSTRUCTION}\nQuery: {text}"
        [embedding] = self._model.encode([instructed], normalize_embeddings=True)
        return list(embedding)
