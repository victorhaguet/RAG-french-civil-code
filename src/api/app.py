"""FastAPI application: query the Code civil corpus and get a grounded answer."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.api.dependencies import get_chat_model, get_store
from src.api.schemas import ChunkOut, QueryRequest, QueryResponse
from src.generation.prompt import render_prompt

app = FastAPI()


def _to_chunk_out(chunk: Document) -> ChunkOut:
    # Every Chunk stored via the ingestion pipeline has a deterministic
    # `{ref}#{n}` id, so a retrieved Document is never id-less in practice.
    assert chunk.id is not None
    return ChunkOut(id=chunk.id, text=chunk.page_content, metadata=chunk.metadata)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    store: Chroma = Depends(get_store),
    chat_model: Any = Depends(get_chat_model),
) -> QueryResponse:
    """Retrieve the most relevant Chunks and generate a grounded answer."""
    chunks = store.similarity_search(request.question, k=request.top_k)
    prompt = render_prompt(question=request.question, chunks=chunks)
    answer = chat_model.invoke(prompt).content

    return QueryResponse(
        answer=answer,
        chunks=[_to_chunk_out(chunk) for chunk in chunks],
    )
