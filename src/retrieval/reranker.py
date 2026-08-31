"""Cross-encoder reranking of the fused Candidate Articles from Hybrid Retrieval.

Reciprocal Rank Fusion combines the Keyword Index and vector index by rank
position alone, with no notion of how relevant a candidate actually is to the
query. The Reranker scores each `(question, Article.texte)` pair directly via
a small multilingual cross-encoder, sharpening precision especially where the
two indexes disagree on ordering.
"""

from __future__ import annotations

from typing import Any

from src.ingestion.dataset import Article

CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker:
    """Reorders candidate Articles by cross-encoder query-article relevance.

    `model` accepts anything duck-typing sentence-transformers'
    `CrossEncoder.predict()` (a real `CrossEncoder`, or a lightweight test
    double).
    """

    def __init__(self, model: Any | None = None) -> None:
        self._model: Any = model if model is not None else self._load_default_model()

    def _load_default_model(self) -> Any:
        # Imported lazily so constructing this class with an injected test
        # double (as every test does) never pays sentence-transformers'
        # heavy import cost.
        from sentence_transformers import CrossEncoder

        return CrossEncoder(CROSS_ENCODER_MODEL)

    def rerank(self, question: str, articles: list[Article]) -> list[Article]:
        """Reorder `articles` by descending cross-encoder relevance to `question`.

        Args:
            question (str): the natural-language query.
            articles (list[Article]): candidate Articles to reorder, in their
                incoming (e.g. fused) order.

        Returns:
            list[Article]: `articles`, reordered most relevant first.
        """
        if not articles:
            return []
        pairs = [(question, article["texte"]) for article in articles]
        scores = self._model.predict(pairs)
        order = sorted(range(len(articles)), key=lambda i: scores[i], reverse=True)
        return [articles[i] for i in order]
