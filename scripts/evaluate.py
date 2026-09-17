"""Score the real RAG pipeline against the eval/ question sets, via `/query`.

Prints RAGAS `faithfulness` and `context_recall` for the Golden Question set, and
binary guardrail-pass rates for the Out-of-Scope Question and Injection Attempt
sets. Safe to run before those files are populated: every metric then reports
"no data".

Usage:
    uv run scripts/evaluate.py
    uv run scripts/evaluate.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse

from src.evaluation.client import build_query_client
from src.evaluation.dataset import load_golden_questions, load_guardrail_questions
from src.evaluation.golden import GoldenReport, evaluate_golden_questions
from src.evaluation.guardrail_report import GuardrailReport, evaluate_guardrail_questions
from src.evaluation.judge import build_judge_metrics

GOLDEN_QUESTIONS_PATH = "eval/golden_questions.jsonl"
OUT_OF_SCOPE_QUESTIONS_PATH = "eval/out_of_scope_questions.jsonl"
INJECTION_ATTEMPTS_PATH = "eval/injection_attempts.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=None,
        help="Live server to evaluate, e.g. http://localhost:8000. "
        "Defaults to an in-process run against the real FastAPI app.",
    )
    args = parser.parse_args()
    client = build_query_client(args.base_url)

    golden_report = evaluate_golden_questions(
        client, load_golden_questions(GOLDEN_QUESTIONS_PATH), build_judge_metrics
    )
    out_of_scope_report = evaluate_guardrail_questions(
        client, load_guardrail_questions(OUT_OF_SCOPE_QUESTIONS_PATH)
    )
    injection_report = evaluate_guardrail_questions(
        client, load_guardrail_questions(INJECTION_ATTEMPTS_PATH)
    )

    _print_golden_report(golden_report)
    print()
    _print_guardrail_report("out_of_scope_guardrail_rate", out_of_scope_report)
    _print_guardrail_report("injection_guardrail_rate", injection_report)


def _print_golden_report(report: GoldenReport) -> None:
    print("Golden Questions")
    if not report.has_data:
        print("  faithfulness: no data")
        print("  context_recall: no data")
        return

    for score in report.scores:
        print(
            f"  - {score.question!r}: "
            f"faithfulness={score.faithfulness:.2f} context_recall={score.context_recall:.2f}"
        )
    print(f"  faithfulness (mean): {report.mean_faithfulness:.2f}")
    print(f"  context_recall (mean): {report.mean_context_recall:.2f}")


def _print_guardrail_report(name: str, report: GuardrailReport) -> None:
    if not report.has_data:
        print(f"{name}: no data")
        return
    print(f"{name}: {report.rate:.2f} ({report.passed}/{report.total})")


if __name__ == "__main__":
    main()
