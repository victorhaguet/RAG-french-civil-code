"""BM25 Keyword Index over Articles' full text, for exact statutory-term matching.

Built lazily in memory from the existing `ArticleStore` — no ingestion
changes, nothing persisted to disk. Indexed over each Article's full `texte`
(not Chunks): BM25 has no fixed-context-window constraint the way embeddings
do, and its own document-length normalization already handles Article-size
variance.
"""

from __future__ import annotations

import re
import threading
from typing import Any

from src.storage.article_store import ArticleStore

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_stemmer: Any | None = None
_stopwords: frozenset[str] | None = None


def _get_stemmer() -> Any:
    global _stemmer
    if _stemmer is None:
        # Imported lazily so importing this module never pays nltk's cost
        # for callers that don't end up using the Keyword Index.
        from nltk.stem.snowball import SnowballStemmer

        _stemmer = SnowballStemmer("french")
    return _stemmer


def _get_stopwords() -> frozenset[str]:
    global _stopwords
    if _stopwords is None:
        import nltk

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        _stopwords = frozenset(stopwords.words("french"))
    return _stopwords


def tokenize(text: str) -> list[str]:
    """Tokenize French text for BM25 indexing and querying.

    Lowercases, splits into word tokens, then applies French Snowball
    stemming and drops French stopwords.

    Args:
        text (str): text to tokenize.

    Returns:
        list[str]: stemmed, stopword-free tokens.
    """
    stemmer = _get_stemmer()
    stop = _get_stopwords()
    words = _TOKEN_RE.findall(text.lower())
    return [stemmer.stem(word) for word in words if word not in stop]


class KeywordIndex:
    """A BM25 index over every in-force Article's full text.

    Built lazily from `article_store` on first `.search()` call, so
    constructing this class stays cheap — no `rank_bm25`/tokenization cost is
    paid until a search is actually made.
    """

    def __init__(self, article_store: ArticleStore) -> None:
        self._article_store = article_store
        self._bm25: Any | None = None
        self._refs: list[str] = []
        # FastAPI runs the sync `/query` endpoint in a thread pool, so
        # concurrent first requests can race to build the index; this lock
        # ensures only one of them actually does.
        self._build_lock = threading.Lock()

    def _ensure_built(self) -> None:
        if self._bm25 is not None:
            return
        with self._build_lock:
            if self._bm25 is not None:
                return
            # Imported lazily so constructing this class never pays
            # rank_bm25's import cost until a search actually needs the
            # index built.
            from rank_bm25 import BM25Okapi

            articles = self._article_store.all()
            refs = [article["ref"] for article in articles]
            corpus = [tokenize(article["texte"]) for article in articles]
            if corpus:
                self._refs = refs
                self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int) -> list[str]:
        """Rank Article refs by BM25 relevance to `query`.

        Args:
            query (str): the natural-language query.
            k (int): the maximum number of refs to return.

        Returns:
            list[str]: up to `k` Article refs with a nonzero BM25 score,
                most relevant first. An Article sharing no term with the
                query is excluded rather than returned as a spurious match.
        """
        self._ensure_built()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self._refs)), key=lambda i: scores[i], reverse=True)
        matched = [i for i in order if scores[i] > 0]
        return [self._refs[i] for i in matched[:k]]
