"""Environment-based configuration, loaded from .env via python-dotenv."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "code_civil")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
