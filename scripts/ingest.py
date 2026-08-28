"""Rebuild the Chroma collection and the Article store from the Code civil dataset.

Usage: uv run scripts/ingest.py
"""

import logging

from src import config
from src.ingestion.pipeline import run_ingestion


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    store = run_ingestion()
    chunk_count = len(store.get()["ids"])
    print(
        f"Ingested {chunk_count} chunks into '{config.CHROMA_COLLECTION_NAME}' "
        f"and the Article store at '{config.ARTICLES_DB_PATH}'."
    )


if __name__ == "__main__":
    main()
