"""SQLite-backed lookup from an Article's `ref` to its full record.

Chroma indexes Chunks for retrieval only; a Chunk's metadata never carries
the Article's full `texte` (see docs/adr/0001-full-articles-in-generation-prompt.md).
This store holds every Article's full record instead, keyed by `ref`, so
retrieval and the generation prompt can each use the right granularity:
Chunks to search, Articles to answer from.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from src.ingestion.dataset import KEPT_FIELDS, Article

_INTEGER_FIELDS = {"dateDebut", "dateFin"}


class ArticleStore:
    """Persists every Article, resolvable by `ref`."""

    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        columns = ", ".join(
            f"{field} {'INTEGER' if field in _INTEGER_FIELDS else 'TEXT'}"
            + (" PRIMARY KEY" if field == "ref" else " NOT NULL")
            for field in KEPT_FIELDS
        )
        with self._connection:
            self._connection.execute(f"CREATE TABLE IF NOT EXISTS articles ({columns})")

    def replace_all(self, articles: Iterable[Article]) -> None:
        """Rebuild the store from scratch with the given Articles.

        Args:
            articles (Iterable[Article]): the full set of Articles to persist.
        """
        placeholders = ", ".join("?" for _ in KEPT_FIELDS)
        rows = [
            tuple(article[field] for field in KEPT_FIELDS)  # type: ignore[literal-required]
            for article in articles
        ]
        with self._connection:
            self._connection.execute("DELETE FROM articles")
            self._connection.executemany(
                f"INSERT INTO articles ({', '.join(KEPT_FIELDS)}) VALUES ({placeholders})",
                rows,
            )

    def get(self, ref: str) -> Article | None:
        """Resolve a `ref` to its full Article.

        Args:
            ref (str): the Article's `ref`.

        Returns:
            Article | None: the Article, or None if no Article has this ref.
        """
        cursor = self._connection.execute(
            f"SELECT {', '.join(KEPT_FIELDS)} FROM articles WHERE ref = ?", (ref,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return cast(Article, dict(zip(KEPT_FIELDS, row, strict=True)))
