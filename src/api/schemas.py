"""Request/response schemas for the query API."""

from __future__ import annotations

from pydantic import BaseModel, Field

DEFAULT_TOP_K = 5


class QueryRequest(BaseModel):
    """A natural-language question, with an optional retrieval size."""

    question: str
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)


class ArticleOut(BaseModel):
    """A Retrieved Article, as cited to the API consumer.

    Carries just enough to display and resolve a citation — the full text
    is fetched separately via `GET /articles/{ref}`.
    """

    ref: str
    sectionParentTitre: str


class QueryResponse(BaseModel):
    """The generated answer, grounded in the Retrieved Articles it cites."""

    answer: str
    articles: list[ArticleOut]


class ArticleDetailOut(BaseModel):
    """A full Article, as returned by `GET /articles/{ref}`."""

    ref: str
    texte: str
    dateDebut: int
    dateFin: int
    etat: str
    version_article: str
    origine: str
    sectionParentTitre: str
