"""Loading the JSONL question sets `scripts/evaluate.py` scores the pipeline against.

See CONTEXT.md's Evaluation section for what a Golden Question, Out-of-Scope Question,
and Injection Attempt each are.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


class GoldenQuestion(TypedDict):
    """A hand-authored question with known-correct ground truth to score retrieval and generation against.

    `reference_article_refs` is part of the Golden Question's ground truth (see
    CONTEXT.md), but isn't fed into scoring: a question can plausibly be answered
    from more than one Article, so a strict set-match against a fixed ref list would
    understate recall. RAGAS's `context_recall` — judged against `reference_answer`,
    not a ref list — is the more representative measure and is what's actually
    scored. Kept in the schema as ground truth for future use (e.g. manual review).
    """

    question: str
    reference_answer: str
    reference_article_refs: list[str]


class GuardrailQuestion(TypedDict):
    """An Out-of-Scope Question or Injection Attempt: just the question, scored as pass/fail."""

    question: str


def load_golden_questions(path: str | Path) -> list[GoldenQuestion]:
    """Load the Golden Question set from a JSONL file, tolerating zero lines.

    Args:
        path (str | Path): path to the JSONL file

    Returns:
        list[GoldenQuestion]: one entry per non-blank line, `[]` for an empty file
    """
    return _load_jsonl(path)


def load_guardrail_questions(path: str | Path) -> list[GuardrailQuestion]:
    """Load an Out-of-Scope Question or Injection Attempt set from a JSONL file, tolerating zero lines.

    Args:
        path (str | Path): path to the JSONL file

    Returns:
        list[GuardrailQuestion]: one entry per non-blank line, `[]` for an empty file
    """
    return _load_jsonl(path)


def _load_jsonl(path: str | Path) -> list[Any]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
