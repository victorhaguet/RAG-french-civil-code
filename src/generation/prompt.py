"""Loading and rendering the Jinja2 RAG prompt template for the query's language."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.ingestion.dataset import DATASET_AS_OF, Article
from src.retrieval.language import QueryLanguage, detect_query_language

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
PROMPT_TEMPLATE_NAMES: dict[QueryLanguage, str] = {
    "fr": "rag_answer_fr.jinja2",
    "en": "rag_answer_en.jinja2",
}

_env = Environment(loader=FileSystemLoader(PROMPT_DIR), autoescape=False)  # noqa: S701


def render_prompt(
    question: str, articles: list[Article], dataset_as_of: str = DATASET_AS_OF
) -> str:
    """Render the RAG prompt template matching the question's detected language.

    Args:
        question (str): the user's natural-language question
        articles (list[Article]): Retrieved Articles to ground the answer in,
            in full — never the Chunks used to find them
        dataset_as_of (str): the Code civil corpus's snapshot date, shown to
            the user when no retrieved Article answers the question

    Returns:
        str: the rendered prompt, ready to send to the chat model
    """
    template_name = PROMPT_TEMPLATE_NAMES[detect_query_language(question)]
    template = _env.get_template(template_name)
    return template.render(question=question, articles=articles, dataset_as_of=dataset_as_of)
