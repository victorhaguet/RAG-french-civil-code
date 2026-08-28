"""Loading and filtering Articles from the Code civil dataset."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypedDict, cast

DATASET_NAME = "louisbrulenaudet/code-civil"
DATASET_SPLIT = "train"
# The dataset has no built-in version/timestamp field, so this is a manually
# tracked snapshot date, updated whenever the corpus is re-ingested.
DATASET_AS_OF = "21 September 2025"

IN_FORCE_ETAT = "VIGUEUR"

# Article fields kept from the source dataset.
KEPT_FIELDS = (
    "ref", # Reference of the article
    "texte", # Text content
    "dateDebut", # Starting date
    "dateFin", # Ending date
    "etat", # State of the law
    "version_article", # Version number of the article
    "origine", # Origin of the document
    "sectionParentTitre", # Title of the parent section
)


class Article(TypedDict):
    """The fields an Article keeps from a raw dataset row."""

    ref: str
    texte: str
    dateDebut: int
    dateFin: int
    etat: str
    version_article: str
    origine: str
    sectionParentTitre: str


def load_raw_rows() -> Iterator[dict]:
    """Load every row of the Code civil dataset from HuggingFace.

    Returns:
        Iterator[dict]: An iterator over the dataset rows, each row
        representing a specific article of the civil code.
    """
    # Imported lazily so tests that pass `raw_rows` directly never need to
    # import `datasets` at all.
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    return iter(dataset)


def to_article(row: dict) -> Article:
    """Project a raw dataset row down to the fields an Article keeps.

    Args:
        row (dict): Dictionary containing one of the extracted rows of the
        dataset.

    Returns:
        Article: Article object of the row 
    """
    return cast(Article, {field: row[field] for field in KEPT_FIELDS})


def is_in_force(article: Article) -> bool:
    """An Article is in force when its etat is VIGUEUR.

    Args:
        article (Article): Article to check

    Returns:
        bool: True if the article parameter 'etat' is equal to 'VIGUEUR'
        False else. 
    """
    return article["etat"] == IN_FORCE_ETAT


def load_articles(raw_rows: Iterable[dict] | None = None) -> list[Article]:
    """Load in-force Articles, projected to their kept fields.

    Pass `raw_rows` to bypass the network call in tests.

    Args:
        raw_rows (Iterable[dict] | None, optional): Raw rows to load. Defaults to None.

    Returns:
        list[Article]: List of articles object
    """
    rows = raw_rows if raw_rows is not None else load_raw_rows()
    return [
        article
        for article in (to_article(row) for row in rows)
        if is_in_force(article)
    ]
