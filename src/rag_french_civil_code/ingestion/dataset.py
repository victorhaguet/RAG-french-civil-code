"""Loading and filtering Articles from the Code civil dataset."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypedDict, cast

DATASET_NAME = "louisbrulenaudet/code-civil"
DATASET_SPLIT = "train"

IN_FORCE_ETAT = "VIGUEUR"

# Article fields kept from the source dataset. `idEliAlias`, `idEli`, `renvoi`
# and `inap` are dropped: confirmed null across every row of the dataset.
KEPT_FIELDS = (
    "ref",
    "texte",
    "dateDebut",
    "dateFin",
    "etat",
    "version_article",
    "origine",
    "sectionParentTitre",
)


class Article(TypedDict):
    ref: str
    texte: str
    dateDebut: int
    dateFin: int
    etat: str
    version_article: str
    origine: str
    sectionParentTitre: str


def load_raw_rows() -> Iterator[dict]:
    """Load every row of the Code civil dataset from HuggingFace."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    return iter(dataset)


def to_article(row: dict) -> Article:
    """Project a raw dataset row down to the fields an Article keeps."""
    return cast(Article, {field: row[field] for field in KEPT_FIELDS})


def is_in_force(article: Article) -> bool:
    """An Article is in force when its etat is VIGUEUR."""
    return article["etat"] == IN_FORCE_ETAT


def load_articles(raw_rows: Iterable[dict] | None = None) -> list[Article]:
    """Load in-force Articles, projected to their kept fields.

    Pass `raw_rows` to bypass the network call in tests.
    """
    rows = raw_rows if raw_rows is not None else load_raw_rows()
    return [
        article
        for article in (to_article(row) for row in rows)
        if is_in_force(article)
    ]
