"""Rebuild the Chroma collection from the Code civil dataset.

Usage: uv run scripts/ingest.py
"""

from rag_french_civil_code import config
from rag_french_civil_code.ingestion.pipeline import run_ingestion


def main() -> None:
    store = run_ingestion()
    chunk_count = len(store.get()["ids"])
    print(f"Ingested {chunk_count} chunks into '{config.CHROMA_COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
