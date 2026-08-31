"""Embeddings backed by a local sentence-transformers model from the E5 family.

Two members of the family are supported, selected via `config.EMBEDDING_MODEL`
(env var `EMBEDDING_MODEL`) or the `model_name` constructor arg:

- `intfloat/multilingual-e5-large-instruct` (the default): documents get no
  prefix; queries get a French/English instruction prefix, chosen via
  `detect_query_language`. Needs a GPU to be fast.
- `intfloat/multilingual-e5-small` (the CPU-friendly fallback): documents and
  queries get a fixed `"passage: "`/`"query: "` prefix, regardless of
  language.

Any other model name is a config error: this repo only knows how to prefix
these two, so supporting a different embedding model family needs a code
change here, not just a new `EMBEDDING_MODEL` value.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from src import config
from src.retrieval.language import detect_query_language

FIXED_PREFIX_MODEL = "intfloat/multilingual-e5-small"
INSTRUCT_MODEL = "intfloat/multilingual-e5-large-instruct"
SUPPORTED_MODELS = (FIXED_PREFIX_MODEL, INSTRUCT_MODEL)

DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "

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

    def __init__(self, model: Any | None = None, model_name: str | None = None) -> None:
        self._model_name = model_name or config.EMBEDDING_MODEL
        if self._model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported EMBEDDING_MODEL {self._model_name!r}. This repo "
                f"only knows how to prefix {SUPPORTED_MODELS!r} — supporting "
                "another embedding model family needs a code change in "
                "src/retrieval/embeddings.py, not just a new env value."
            )
        self._model: Any = model if model is not None else self._load_default_model()

    def _load_default_model(self) -> Any:
        # Imported lazily so constructing this class with an injected test
        # double (as every test does) never pays sentence-transformers'
        # heavy import cost.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self._model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed documents into vectors

        Args:
            texts (list[str]): Texts to embed

        Returns:
            list[list[float]]: List of obtained vectors
        """
        prefixed = (
            texts
            if self._model_name == INSTRUCT_MODEL
            else [DOCUMENT_PREFIX + text for text in texts]
        )
        embeddings = self._model.encode(prefixed, normalize_embeddings=True)
        return [[float(x) for x in vector] for vector in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query.

        The query is prefixed according to `self._model_name`'s convention:
        an instruction (adapted to the query's detected language) for
        `INSTRUCT_MODEL`, or the fixed `"query: "` prefix otherwise.

        Args:
            text (str): Query to embed

        Returns:
            list[float]: Vector obtained
        """
        if self._model_name == INSTRUCT_MODEL:
            instruction = QUERY_INSTRUCTIONS[detect_query_language(text)]
            prefixed = f"Instruct: {instruction}\nQuery: {text}"
        else:
            prefixed = QUERY_PREFIX + text
        [embedding] = self._model.encode([prefixed], normalize_embeddings=True)
        return [float(x) for x in embedding]
