"""Rebuild the Chroma collection and the Article store from the Code civil dataset.

Usage: uv run scripts/ingest.py
"""

from src import config
from src.ingestion.pipeline import run_ingestion


def main() -> None:
    store = run_ingestion()
    chunk_count = len(store.get()["ids"])
    print(
        f"Ingested {chunk_count} chunks into '{config.CHROMA_COLLECTION_NAME}' "
        f"and the Article store at '{config.ARTICLES_DB_PATH}'."
    )


if __name__ == "__main__":
    main()
