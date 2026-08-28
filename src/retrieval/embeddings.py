"""Embeddings for multilingual-e5-small.

The model is asymmetric: document/passage text is embedded with a
`"passage: "` prefix, while queries get a `"query: "` prefix — the
convention this model (and the wider E5 family) was trained on. Unlike the
`-instruct` E5 variants, these prefixes are fixed and don't vary with the
query's language.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

MODEL_NAME = "intfloat/multilingual-e5-small"

DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "


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
        # Imported lazily so constructing this class with an injected test
        # double (as every test does) never pays sentence-transformers'
        # heavy import cost.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed documents into vectors

        Args:
            texts (list[str]): Texts to embed

        Returns:
            list[list[float]]: List of obtained vectors
        """
        prefixed = [DOCUMENT_PREFIX + text for text in texts]
        embeddings = self._model.encode(prefixed, normalize_embeddings=True)
        return [[float(x) for x in vector] for vector in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query.

        The query is embedded with the `"query: "` prefix this model
        expects (specific to intfloat/multilingual-e5-small).

        Args:
            text (str): Query to embed

        Returns:
            list[float]: Vector obtained
        """
        [embedding] = self._model.encode([QUERY_PREFIX + text], normalize_embeddings=True)
        return [float(x) for x in embedding]
