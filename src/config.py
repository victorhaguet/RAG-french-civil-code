"""Environment-based configuration, loaded from .env via python-dotenv."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "code_civil")

ARTICLES_DB_PATH = os.getenv("ARTICLES_DB_PATH", "data/articles.db")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Number of Articles `/query` returns when the caller doesn't pass `top_k`.
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# Hybrid retrieval: each index fetches `max(FETCH_K_MULTIPLIER * top_k, MIN_FETCH_K)`
# candidates before fusion, so the fused pool is meaningfully larger than top_k.
FETCH_K_MULTIPLIER = 4
MIN_FETCH_K = 20

# Weighted Reciprocal Rank Fusion: per-index trust, tunable independently
# without touching fusion logic itself.
RRF_K = 60
RRF_WEIGHT_BM25 = 1.0
RRF_WEIGHT_VECTOR = 1.0
