"""The RAGAS metric shape `src/evaluation/golden.py` scores Golden Questions against."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class MetricScore(Protocol):
    """Duck-types the result of a RAGAS metric's `.score()` (RAGAS's `MetricResult`)."""

    @property
    def value(self) -> float:
        """The metric score in [0, 1]."""
        ...


class ScorableMetric(Protocol):
    """Duck-types RAGAS's `ragas.metrics.collections` metrics (`Faithfulness`, `ContextRecall`, ...).

    Typed as a Protocol, not the real RAGAS classes, so `src/evaluation/golden.py`
    never has to import `ragas` (a heavy, lazily-imported dependency — see
    `src/evaluation/judge.py`) just to type-check against it. Tests inject a
    lightweight double instead of a real metric.
    """

    def score(self, **kwargs: Any) -> MetricScore:
        """Score one sample; the result's `.value` is the metric score in [0, 1]."""
        ...


@dataclass(frozen=True)
class JudgeMetrics:
    """The two RAGAS metrics every Golden Question is scored against, bound to the judge LLM."""

    faithfulness: ScorableMetric
    context_recall: ScorableMetric
