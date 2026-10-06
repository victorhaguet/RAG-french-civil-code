# Self-contained, CPU-only image of the query API: the Article store, the
# Chroma vector store and every model (embedder, Reranker, NLTK stopwords) are
# built/downloaded here at build time, so the container never downloads
# anything at runtime. No secret is needed to build it -- ingestion only
# embeds, it never calls OpenAI.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /bin/uv

# Fixed at build time: the baked vector store and the runtime embedder must
# use the same model (Chroma stores fixed-size vectors per collection). The
# compose file pins it again at runtime, so a local `.env` can't override it.
# e5-small rather than the default -large-instruct: this image is CPU-only.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
# Must match src/retrieval/reranker.py's CROSS_ENCODER_MODEL -- downloaded
# before the source is copied in, so a source change doesn't re-download it.
# The offline check at the end of this file fails the build on a mismatch.
ARG RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    HF_HOME=/opt/models/huggingface \
    NLTK_DATA=/opt/models/nltk_data \
    CHROMA_PERSIST_DIR=/app/data/chroma \
    ARTICLES_DB_PATH=/app/data/articles.db \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /opt/models \
    && chown app:app /app /app/data /opt/models

WORKDIR /app

# Dependencies first (cpu torch extra, no dev/eval groups), in their own layer
# so source changes don't reinstall them.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --extra cpu --no-dev --no-install-project

USER app

# Models, before the source for the same reason.
RUN python -m nltk.downloader -d "$NLTK_DATA" stopwords \
    && python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANKER_MODEL}')"

# Only the code ingestion runs, so editing the API, the prompt or the rest of
# the retrieval code doesn't invalidate the baked data (re-ingesting means
# re-embedding every Article on CPU). Not installed as a package: WORKDIR
# /app is on sys.path for `python -m`/uvicorn, which is all `src` needs.
COPY --chown=app:app src/__init__.py src/config.py ./src/
COPY --chown=app:app src/ingestion ./src/ingestion
COPY --chown=app:app src/storage ./src/storage
COPY --chown=app:app src/retrieval/__init__.py src/retrieval/embeddings.py ./src/retrieval/
COPY --chown=app:app scripts/ingest.py ./scripts/

# The raw dataset is only read here: drop its caches in the same layer so the
# image doesn't carry them (the embedder downloaded above stays).
RUN python -m scripts.ingest \
    && rm -rf "$HF_HOME/datasets" "$HF_HOME/hub/datasets--louisbrulenaudet--code-civil"

COPY --chown=app:app src ./src
COPY --chown=app:app prompts ./prompts

# From here on nothing may be downloaded: check every model the API loads
# resolves from the baked caches alone.
ENV HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1
RUN python -c "from src.retrieval.embeddings import MultilingualE5Embeddings; \
from src.retrieval.reranker import Reranker; \
from src.retrieval.keyword_index import _get_stopwords; \
MultilingualE5Embeddings(); Reranker(); _get_stopwords()"

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
