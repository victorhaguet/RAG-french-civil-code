"""Splitting Articles into Chunks and building vectorstore-ready Documents."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.dataset import KEPT_FIELDS, Article

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Metadata fields carried onto every Chunk: every kept Article field except
# `texte`, which becomes the Chunk's embedded page content instead. Derived
# from KEPT_FIELDS so the two stay in sync automatically.
METADATA_FIELDS = tuple(field for field in KEPT_FIELDS if field != "texte")

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


def chunk_article(article: Article) -> list[Document]:
    """Split one Article's text into Chunks, as vectorstore Documents.

    An Article whose text fits in one chunk yields a single Document
    (`{ref}#0`); a longer one yields several (`{ref}#0`, `{ref}#1`, ...),
    each carrying its own copy of the Article's metadata.

    Args:
        article (Article): article to split

    Returns:
        list[Document]: list of chunks extracted from the article
    """
    chunks_text = _splitter.split_text(article["texte"])
    return [
        Document(
            page_content=chunk_text,
            metadata={field: article[field] for field in METADATA_FIELDS},  # type: ignore[literal-required]
            id=f"{article['ref']}#{n}",
        )
        for n, chunk_text in enumerate(chunks_text)
    ]


def build_documents(articles: Iterable[Article]) -> list[Document]:
    """Chunk every Article into the full list of vectorstore Documents.

    Args:
        articles (Iterable[Article]): articles to transform to chunks

    Returns:
        list[Document]: List of chunks (ready to be embedded)
    """
    documents: list[Document] = []
    for article in articles:
        documents.extend(chunk_article(article))
    return documents
