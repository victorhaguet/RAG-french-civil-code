"""Environment-based configuration, loaded from .env via python-dotenv."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "code_civil")
