"""Loading and rendering the Jinja2 RAG prompt template."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.ingestion.dataset import DATASET_AS_OF, Article

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
PROMPT_TEMPLATE_NAME = "rag_answer_fr.jinja2"

_env = Environment(loader=FileSystemLoader(PROMPT_DIR), autoescape=False)  # noqa: S701


def render_prompt(
    question: str, articles: list[Article], dataset_as_of: str = DATASET_AS_OF
) -> str:
    """Render the French RAG prompt template.

    Only French questions reach this point: `/query` refuses any other
    language before retrieval.

    Args:
        question (str): the user's natural-language question
        articles (list[Article]): Retrieved Articles to ground the answer in,
            in full — never the Chunks used to find them
        dataset_as_of (str): the Code civil corpus's snapshot date, shown to
            the user when no retrieved Article answers the question

    Returns:
        str: the rendered prompt, ready to send to the chat model
    """
    template = _env.get_template(PROMPT_TEMPLATE_NAME)
    return template.render(question=question, articles=articles, dataset_as_of=dataset_as_of)
