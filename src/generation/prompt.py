"""Loading and rendering the Jinja2 RAG prompt template for the query's language."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from langchain_core.documents import Document

from src.retrieval.language import QueryLanguage, detect_query_language

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
PROMPT_TEMPLATE_NAMES: dict[QueryLanguage, str] = {
    "fr": "rag_answer_fr.jinja2",
    "en": "rag_answer_en.jinja2",
}

_env = Environment(loader=FileSystemLoader(PROMPT_DIR), autoescape=False)  # noqa: S701


def render_prompt(question: str, chunks: list[Document]) -> str:
    """Render the RAG prompt template matching the question's detected language.

    Args:
        question (str): the user's natural-language question
        chunks (list[Document]): retrieved Chunks to ground the answer in

    Returns:
        str: the rendered prompt, ready to send to the chat model
    """
    template_name = PROMPT_TEMPLATE_NAMES[detect_query_language(question)]
    template = _env.get_template(template_name)
    return template.render(question=question, chunks=chunks)
