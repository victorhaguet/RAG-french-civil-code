import sys

import pytest

from src.evaluation.judge import _stub_langchain_community_vertexai

_MODULE_NAME = "langchain_community.chat_models.vertexai"


@pytest.fixture(autouse=True)
def _clear_stub():
    sys.modules.pop(_MODULE_NAME, None)
    yield
    sys.modules.pop(_MODULE_NAME, None)


def test_registers_a_stub_module_satisfying_ragas_import() -> None:
    _stub_langchain_community_vertexai()

    stub = sys.modules[_MODULE_NAME]
    with pytest.raises(RuntimeError, match="Vertex AI"):
        stub.ChatVertexAI()


def test_is_idempotent_and_never_overwrites_a_real_module() -> None:
    sentinel = object()
    sys.modules[_MODULE_NAME] = sentinel  # type: ignore[assignment]

    _stub_langchain_community_vertexai()

    assert sys.modules[_MODULE_NAME] is sentinel
