"""Scoring Out-of-Scope Questions and Injection Attempts: a binary guardrail-pass rate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.evaluation.client import QueryClient
from src.evaluation.dataset import GuardrailQuestion
from src.evaluation.guardrail import is_out_of_scope_answer


@dataclass
class GuardrailReport:
    """An Out-of-Scope Question or Injection Attempt set's pass rate.

    `has_data` is False for an empty set — `rate` is then `None` ("no data"),
    never 0 or 1.
    """

    total: int
    passed: int

    @property
    def has_data(self) -> bool:
        return self.total > 0

    @property
    def rate(self) -> float | None:
        return self.passed / self.total if self.has_data else None


def evaluate_guardrail_questions(
    client: QueryClient, questions: Sequence[GuardrailQuestion]
) -> GuardrailReport:
    """Score every question by driving the real pipeline through `/query`.

    A question passes when its answer takes the Out-of-Scope Answer shape (see
    `src.evaluation.guardrail.is_out_of_scope_answer`) — the same check applies to
    both Out-of-Scope Questions and Injection Attempts, per the issue's acceptance
    criteria; only the aggregate each set is reported under differs.

    Args:
        client (QueryClient): drives `/query`
        questions (Sequence[GuardrailQuestion]): an Out-of-Scope Question or
            Injection Attempt set

    Returns:
        GuardrailReport: pass count out of total; `total=0` for an empty set
    """
    passed = 0
    for item in questions:
        response = client.post("/query", json={"question": item["question"]})
        response.raise_for_status()
        if is_out_of_scope_answer(response.json()["answer"]):
            passed += 1

    return GuardrailReport(total=len(questions), passed=passed)
