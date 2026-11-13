"""Request/response schemas for the query API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from src import config


class AboutResponse(BaseModel):
    """About: the facts the Web UI's header presents, read from the running server."""

    dataset_as_of: date = Field(description="The Code civil dataset's snapshot date.")
    article_count: int = Field(description="How many Articles the Article store holds.")
    model: str = Field(description="The configured generation model.")


class QueryRequest(BaseModel):
    """A natural-language question, with an optional retrieval size."""

    question: str
    top_k: int = Field(default=config.DEFAULT_TOP_K, gt=0)


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


class ArticlesMessage(BaseModel):
    """A Streamed Answer's first message: the Retrieved Articles, before any of the answer."""

    type: Literal["articles"] = "articles"
    articles: list[ArticleOut]


class DeltaMessage(BaseModel):
    """The next piece of a Streamed Answer's answer, in order."""

    type: Literal["delta"] = "delta"
    text: str


class DoneMessage(BaseModel):
    """A Streamed Answer's last message, when the whole answer was sent."""

    type: Literal["done"] = "done"


class ErrorMessage(BaseModel):
    """A Streamed Answer's last message, instead of `done`, when generation failed midway.

    Carries a generic message only, never the failure's internal details.
    """

    type: Literal["error"] = "error"
    message: str
