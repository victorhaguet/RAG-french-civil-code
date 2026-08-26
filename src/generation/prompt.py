"""Loading and rendering the fixed Jinja2 RAG prompt template."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from langchain_core.documents import Document

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
PROMPT_TEMPLATE_NAME = "rag_answer.jinja2"

_env = Environment(loader=FileSystemLoader(PROMPT_DIR), autoescape=False)  # noqa: S701


def render_prompt(question: str, chunks: list[Document]) -> str:
    """Render the fixed RAG prompt template with a question and its Chunks.

    Args:
        question (str): the user's natural-language question
        chunks (list[Document]): retrieved Chunks to ground the answer in

    Returns:
        str: the rendered prompt, ready to send to the chat model
    """
    template = _env.get_template(PROMPT_TEMPLATE_NAME)
    return template.render(question=question, chunks=chunks)
