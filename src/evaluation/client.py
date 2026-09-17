"""Building the client that drives the real pipeline via `/query` and `/articles/{ref}`.

A local run (in-process, the real, unfaked `src/api/dependencies.py` wiring) and a
live run (HTTP against a running server) share one call surface — `.post()`/`.get()`
returning a response with `.raise_for_status()` and `.json()` — so callers never
reimplement retrieval/fusion/rerank/generation in parallel. `TestClient` and
`httpx.Client` satisfy this structurally, not by a shared base class: Starlette's
`TestClient` wraps its own vendored `httpx`-alike, not the `httpx` package this
project depends on.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from fastapi.testclient import TestClient

from src.api.app import app


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
    """
    if base_url is not None:
        return httpx.Client(base_url=base_url)
    return TestClient(app)
