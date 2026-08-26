"""Embeddings for multilingual-e5-large-instruct.

The model is asymmetric: document/passage text is embedded with no
instruction prefix, while queries need one (`"Instruct: {task}\\nQuery:
{query}"`) to retrieve well. The instruction text is adapted to the query's
detected language (French/English, French fallback), via Lingua.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from src.retrieval.language import detect_query_language

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

QUERY_INSTRUCTIONS = {
    "fr": (
        "Étant donné une question juridique, retrouve les articles du Code "
        "civil pertinents pour y répondre."
    ),
    "en": (
        "Given a legal question, retrieve the Code civil articles relevant "
        "to answering it."
    ),
}


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
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [list(vector) for vector in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query.

        The query is embedded with an instruction prefix (specific to
        intfloat/multilingual-e5-large-instruct) adapted to the query's
        detected language.

        Args:
            text (str): Query to embed

        Returns:
            list[float]: Vector obtained
        """
        instruction = QUERY_INSTRUCTIONS[detect_query_language(text)]
        instructed = f"Instruct: {instruction}\nQuery: {text}"
        [embedding] = self._model.encode([instructed], normalize_embeddings=True)
        return list(embedding)
