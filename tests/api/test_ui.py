"""The Web UI's routes: `/` redirects to `/ui/`, which serves the built UI without a token.

Served from a fixture output (`tests/fixtures/ui`), so these tests don't need
the Node build.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app

UI_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "ui"


def _client() -> TestClient:
    return TestClient(create_app(ui_dir=UI_FIXTURE_DIR))


def test_root_redirects_to_the_web_ui() -> None:
    response = _client().get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"


def test_web_ui_is_served_without_a_token() -> None:
    response = _client().get("/ui/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Web UI fixture" in response.text


def test_web_ui_without_trailing_slash_leads_to_the_web_ui() -> None:
    response = _client().get("/ui")

    assert response.status_code == 200
    assert "Web UI fixture" in response.text


def test_web_ui_missing_file_is_a_404() -> None:
    response = _client().get("/ui/missing.js")

    assert response.status_code == 404


def test_mcp_route_is_still_served_after_the_web_ui_routes() -> None:
    # The MCP app is mounted at the root, after the Web UI's routes: `/mcp`
    # must still reach it (a 401 asking for a token), not a 404.
    response = _client().post("/mcp", json={})

    assert response.status_code == 401
