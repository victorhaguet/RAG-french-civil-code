"""Scoring Golden Questions: RAGAS `faithfulness` and `context_recall`, per-question and aggregate."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.evaluation.client import QueryClient
from src.evaluation.dataset import GoldenQuestion
from src.evaluation.metrics import ScorableMetric


@dataclass
class GoldenQuestionScore:
    """One Golden Question's scores, plus enough of the run to review by hand.

    `answer` and `articles` are the real pipeline's output for this question;
    `reference_answer` and `reference_article_refs` are its ground truth
    (see `GoldenQuestion`) -- carried alongside for `src.evaluation.report`
    to show expected vs. actual, not used in scoring itself.

    `faithfulness`/`context_recall` are `None` when the judge LLM itself failed
    to score this question (see `_safe_judge_score`) -- the `/query` call that
    produced `answer`/`articles` above already succeeded; only that one judge
    verdict is missing. `GoldenReport`'s means exclude a `None`, never count it
    as 0.
    """

    question: str
    answer: str
    articles: list[str]
    reference_answer: str
    reference_article_refs: list[str]
    faithfulness: float | None
    context_recall: float | None


@dataclass
class GoldenReport:
    """The Golden Question set's scores, per-question and aggregate.

    `has_data` is False for an empty Golden Question set — `mean_faithfulness`
    and `mean_context_recall` are then `None` ("no data"), never 0. They're
    also `None` (not 0) if every question's judge call for that metric failed
    -- `faithfulness_scored`/`context_recall_scored` say how many actually did.
    """

    scores: list[GoldenQuestionScore] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return bool(self.scores)

    @property
    def mean_faithfulness(self) -> float | None:
        return _mean(score.faithfulness for score in self.scores if score.faithfulness is not None)

    @property
    def mean_context_recall(self) -> float | None:
        return _mean(score.context_recall for score in self.scores if score.context_recall is not None)

    @property
    def faithfulness_scored(self) -> int:
        return sum(1 for score in self.scores if score.faithfulness is not None)

    @property
    def context_recall_scored(self) -> int:
        return sum(1 for score in self.scores if score.context_recall is not None)


def evaluate_golden_questions(
    client: QueryClient,
    questions: Sequence[GoldenQuestion],
    build_metrics: Callable[[], tuple[ScorableMetric, ScorableMetric]],
) -> GoldenReport:
    """Score every Golden Question by driving the real pipeline through `/query`.

    `build_metrics` (typically `src.evaluation.judge.build_judge_metrics`) is only
    called when `questions` is non-empty, so scoring an empty set never needs a
    judge LLM.

    Args:
        client (QueryClient): drives `/query` and `/articles/{ref}`
        questions (Sequence[GoldenQuestion]): the Golden Question set
        build_metrics (Callable[[], tuple[ScorableMetric, ScorableMetric]]):
            builds (faithfulness, context_recall), bound to the judge LLM

    Returns:
        GoldenReport: per-question and aggregate scores; empty for an empty set
    """
    if not questions:
        return GoldenReport()

    faithfulness_metric, context_recall_metric = build_metrics()

    scores = [
        _score_golden_question(client, golden, faithfulness_metric, context_recall_metric)
        for golden in questions
    ]
    return GoldenReport(scores=scores)


def _score_golden_question(
    client: QueryClient,
    golden: GoldenQuestion,
    faithfulness_metric: ScorableMetric,
    context_recall_metric: ScorableMetric,
) -> GoldenQuestionScore:
    response = client.post("/query", json={"question": golden["question"]})
    response.raise_for_status()
    body = response.json()

    contexts = [_fetch_article_text(client, article["ref"]) for article in body["articles"]]

    faithfulness_score = _safe_judge_score(
        golden["question"],
        "faithfulness",
        faithfulness_metric,
        user_input=golden["question"],
        response=body["answer"],
        retrieved_contexts=contexts,
    )
    context_recall_score = _safe_judge_score(
        golden["question"],
        "context_recall",
        context_recall_metric,
        user_input=golden["question"],
        retrieved_contexts=contexts,
        reference=golden["reference_answer"],
    )

    return GoldenQuestionScore(
        question=golden["question"],
        answer=body["answer"],
        articles=[article["ref"] for article in body["articles"]],
        reference_answer=golden["reference_answer"],
        reference_article_refs=golden["reference_article_refs"],
        faithfulness=faithfulness_score,
        context_recall=context_recall_score,
    )


def _safe_judge_score(question: str, metric_name: str, metric: ScorableMetric, **kwargs: Any) -> float | None:
    """Score one sample, tolerating the judge LLM itself failing.

    The judge is an external LLM call and can fail on its own -- rate limits, or
    a verbose/long answer whose structured-output verdict gets truncated
    mid-JSON (see `src.evaluation.judge`'s `_JUDGE_MAX_TOKENS`) -- independently
    of whether this pipeline's own `/query` call (already succeeded by the time
    this runs) has a bug. Returns `None` rather than raising, so one bad judge
    call doesn't abort the whole Golden Question run.
    """
    try:
        return metric.score(**kwargs).value
    except Exception as exc:
        print(f"  ! {metric_name} judge error on {question!r}: {exc}", file=sys.stderr)
        return None


def _fetch_article_text(client: QueryClient, ref: str) -> str:
    response = client.get(f"/articles/{ref}")
    response.raise_for_status()
    text: str = response.json()["texte"]
    return text


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None
