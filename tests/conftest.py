from collections.abc import Iterator

import pytest

from src.api.app import app


@pytest.fixture(autouse=True)
def _clear_dependency_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()
