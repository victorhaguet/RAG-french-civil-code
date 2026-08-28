"""Building the OpenAI-compatible chat model used for answer generation."""

from __future__ import annotations

from typing import Any

from src import config


def build_chat_model() -> Any:
    """Build a `ChatOpenAI` client, fully configured via environment variables.

    No provider is hardcoded: base URL, API key and model all come from
    `config`, so this can point at OpenAI itself or any OpenAI-compatible
    endpoint.

    Returns:
        Any: a `ChatOpenAI` instance
    """
    # Imported lazily so building the FastAPI app (and every test that
    # overrides this dependency) never pays langchain-openai's import cost.
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    api_key = SecretStr(config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
    return ChatOpenAI(
        base_url=config.OPENAI_BASE_URL,
        api_key=api_key,
        model=config.OPENAI_MODEL,
    )
