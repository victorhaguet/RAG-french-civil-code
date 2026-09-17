"""Loading the JSONL question sets `scripts/evaluate.py` scores the pipeline against.

See CONTEXT.md's Evaluation section for what a Golden Question, Out-of-Scope Question,
and Injection Attempt each are.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class GoldenQuestion(TypedDict):
    """A hand-authored question with known-correct ground truth to score retrieval and generation against."""

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
    return [json.loads(line) for line in _read_nonblank_lines(path)]


def load_guardrail_questions(path: str | Path) -> list[GuardrailQuestion]:
    """Load an Out-of-Scope Question or Injection Attempt set from a JSONL file, tolerating zero lines.

    Args:
        path (str | Path): path to the JSONL file

    Returns:
        list[GuardrailQuestion]: one entry per non-blank line, `[]` for an empty file
    """
    return [json.loads(line) for line in _read_nonblank_lines(path)]


def _read_nonblank_lines(path: str | Path) -> list[str]:
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
