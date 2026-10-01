"""The RAGAS metric shape `src/evaluation/golden.py` scores Golden Questions against."""

from __future__ import annotations

from typing import Any, Protocol


class ScorableMetric(Protocol):
    """Duck-types RAGAS's `ragas.metrics.collections` metrics (`Faithfulness`, `ContextRecall`, ...).

    Typed as a Protocol, not the real RAGAS classes, so `src/evaluation/golden.py`
    never has to import `ragas` (a heavy, lazily-imported dependency — see
    `src/evaluation/judge.py`) just to type-check against it. Tests inject a
    lightweight double instead of a real metric.
    """

    def score(self, **kwargs: Any) -> Any:
        """Score one sample; the result's `.value` is the metric score in [0, 1]."""
        ...
