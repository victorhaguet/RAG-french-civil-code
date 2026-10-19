"""Building the client that drives the real pipeline via `/query` and `/articles/{ref}`.

A local run (in-process, the real, unfaked `src/api/dependencies.py` wiring) and a
live run (HTTP against a running server) share one call surface — `.post()`/`.get()`
returning a response with `.raise_for_status()` and `.json()` — so callers never
reimplement retrieval/fusion/rerank/generation in parallel. `TestClient` and
`httpx.Client` satisfy this structurally, not by a shared base class: Starlette's
`TestClient` wraps its own vendored `httpx`-alike, not the `httpx` package this
project depends on.

A live run authenticates with the Access Token in `EVAL_ACCESS_TOKEN`. An
in-process run has no token and needs none: like the stdio MCP Server, it runs
in the same process as the app, so it calls the protected routes as a fixed
trusted identity.
"""

from __future__ import annotations

import sys
from typing import Any, Protocol

import httpx
from fastapi.testclient import TestClient

from src import config
from src.api.app import app
from src.api.dependencies import require_access_token
from src.auth.tokens import QUERY_ROLE, Identity

# httpx.Client's own default (5s, applied separately to connect/write/read/pool)
# is far too short for /query: a single call does real Hybrid Retrieval
# (embedding + BM25), cross-encoder reranking, and a real network round-trip to
# the OpenAI API for generation, any one of which can eat the whole budget on
# its own -- routinely leading to a spurious httpx.ReadTimeout on an otherwise
# fine, just-slow request. 60s comfortably covers a cold request.
_LIVE_CLIENT_TIMEOUT = 60.0

_IN_PROCESS_IDENTITY = Identity(
    subject="in-process-evaluation",
    username="in-process-evaluation",
    roles=frozenset({QUERY_ROLE}),  # `/query` requires it
    expires_at=sys.maxsize,  # never expires
)


class MissingAccessTokenError(RuntimeError):
    """A live run was asked for without an `EVAL_ACCESS_TOKEN` to call it with."""


class QueryClient(Protocol):
    """The `.post`/`.get` call surface `src/evaluation/golden.py` and
    `src/evaluation/guardrail_report.py` drive the pipeline through."""

    def post(self, url: str, *, json: dict[str, Any]) -> Any: ...

    def get(self, url: str) -> Any: ...


def build_query_client(base_url: str | None) -> QueryClient:
    """Build the client `scripts/evaluate.py` drives the pipeline through.

    Args:
        base_url (str | None): a live server URL (e.g. `http://localhost:8000`),
            or `None` to run in-process against `src.api.app.app` directly

    Returns:
        QueryClient: an `httpx.Client` (live) or `TestClient` (in-process)

    Raises:
        MissingAccessTokenError: for a live run without `EVAL_ACCESS_TOKEN`,
            which would otherwise get a `401` on every single question
    """
    if base_url is not None:
        if not config.EVAL_ACCESS_TOKEN:
            raise MissingAccessTokenError(
                f"Set EVAL_ACCESS_TOKEN to an Access Token to evaluate {base_url}: "
                "its /query and /articles/{ref} routes require one."
            )
        return httpx.Client(
            base_url=base_url,
            timeout=_LIVE_CLIENT_TIMEOUT,
            headers={"Authorization": f"Bearer {config.EVAL_ACCESS_TOKEN}"},
        )
    # Overrides the shared module-level `app` for the rest of the process: fine
    # for scripts/evaluate.py, which serves nothing else from it.
    app.dependency_overrides[require_access_token] = lambda: _IN_PROCESS_IDENTITY
    return TestClient(app)
