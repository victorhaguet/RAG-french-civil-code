"""Request/response schemas for the query API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

DEFAULT_TOP_K = 5


class QueryRequest(BaseModel):
    """A natural-language question, with an optional retrieval size."""

    question: str
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)


class ChunkOut(BaseModel):
    """A retrieved Chunk, as returned to the API consumer."""

    id: str
    text: str
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    """The generated answer, grounded in the Chunks it was based on."""

    answer: str
    chunks: list[ChunkOut]
