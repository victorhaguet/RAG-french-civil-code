"""Scoring Golden Questions: RAGAS `faithfulness` and `context_recall`, per-question and aggregate."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from src.evaluation.client import QueryClient
from src.evaluation.dataset import GoldenQuestion
from src.evaluation.metrics import ScorableMetric


@dataclass
class GoldenQuestionScore:
    """One Golden Question's scores."""

    question: str
    faithfulness: float
    context_recall: float


@dataclass
class GoldenReport:
    """The Golden Question set's scores, per-question and aggregate.

    `has_data` is False for an empty Golden Question set — `mean_faithfulness`
    and `mean_context_recall` are then `None` ("no data"), never 0.
    """

    scores: list[GoldenQuestionScore] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return bool(self.scores)

    @property
    def mean_faithfulness(self) -> float | None:
        return _mean(score.faithfulness for score in self.scores) if self.has_data else None

    @property
    def mean_context_recall(self) -> float | None:
        return _mean(score.context_recall for score in self.scores) if self.has_data else None


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

    faithfulness_score = faithfulness_metric.score(
        user_input=golden["question"],
        response=body["answer"],
        retrieved_contexts=contexts,
    ).value
    context_recall_score = context_recall_metric.score(
        user_input=golden["question"],
        retrieved_contexts=contexts,
        reference=golden["reference_answer"],
    ).value

    return GoldenQuestionScore(
        question=golden["question"],
        faithfulness=faithfulness_score,
        context_recall=context_recall_score,
    )


def _fetch_article_text(client: QueryClient, ref: str) -> str:
    response = client.get(f"/articles/{ref}")
    response.raise_for_status()
    text: str = response.json()["texte"]
    return text


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)
