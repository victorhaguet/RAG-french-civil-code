"""Score the real RAG pipeline against the eval/ question sets, via `/query`.

Prints RAGAS `faithfulness` and `context_recall` for the Golden Question set, and
binary guardrail-pass rates for the Out-of-Scope Question and Injection Attempt
sets. Also writes a full Markdown report -- every question's own answer, retrieved
articles, and score/pass-fail, not just the aggregates -- to `eval/results.md`.
Safe to run before those files are populated: every metric then reports "no data".

Three modes (`--mode`):
    report (default)  Print the scores and exit 0.
    gate               Also compare the scores against the committed
                        `eval/baseline.json` (via `src.evaluation.gate`) and exit
                        non-zero, printing the regressed metric(s), on failure.
    update-baseline    Overwrite `eval/baseline.json` with the just-computed
                        scores. Intended to be run only after a passing `gate` run.

Usage:
    uv run scripts/evaluate.py
    uv run scripts/evaluate.py --base-url http://localhost:8000
    uv run scripts/evaluate.py --mode gate
    uv run scripts/evaluate.py --mode update-baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.evaluation.client import build_query_client
from src.evaluation.dataset import load_golden_questions, load_guardrail_questions
from src.evaluation.gate import Scores, evaluate_gate
from src.evaluation.golden import GoldenReport, evaluate_golden_questions
from src.evaluation.guardrail_report import GuardrailReport, evaluate_guardrail_questions
from src.evaluation.judge import build_judge_metrics
from src.evaluation.report import render_markdown_report

logger = logging.getLogger(__name__)

GOLDEN_QUESTIONS_PATH = "eval/golden_questions.jsonl"
OUT_OF_SCOPE_QUESTIONS_PATH = "eval/out_of_scope_questions.jsonl"
INJECTION_ATTEMPTS_PATH = "eval/injection_attempts.jsonl"
BASELINE_PATH = "eval/baseline.json"
RESULTS_REPORT_PATH = "eval/results.md"


def main() -> None:
    # INFO on the root logger so every module's logger.info(...) -- this
    # script's own progress, and the full /query pipeline's (src.api.app,
    # src.retrieval.keyword_index, ...) -- surfaces here, with a timestamp:
    # the key piece of information to diagnose where and how long a run is
    # stuck, not just that it is.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=None,
        help="Live server to evaluate, e.g. http://localhost:8000. "
        "Defaults to an in-process run against the real FastAPI app.",
    )
    parser.add_argument(
        "--mode",
        choices=["report", "gate", "update-baseline"],
        default="report",
        help="'report' (default) prints the scores and exits 0. 'gate' also compares "
        "them against eval/baseline.json and exits non-zero on regression. "
        "'update-baseline' overwrites eval/baseline.json with the computed scores "
        "-- run only after a passing 'gate' run.",
    )
    args = parser.parse_args()
    client = build_query_client(args.base_url)

    logger.info("Scoring Golden Questions...")
    golden_report = evaluate_golden_questions(
        client, load_golden_questions(GOLDEN_QUESTIONS_PATH), build_judge_metrics
    )
    logger.info("Scoring Out-of-Scope Questions...")
    out_of_scope_report = evaluate_guardrail_questions(
        client, load_guardrail_questions(OUT_OF_SCOPE_QUESTIONS_PATH)
    )
    logger.info("Scoring Injection Attempts...")
    injection_report = evaluate_guardrail_questions(
        client, load_guardrail_questions(INJECTION_ATTEMPTS_PATH)
    )

    _print_golden_report(golden_report)
    print()
    _print_guardrail_report("out_of_scope_guardrail_rate", out_of_scope_report)
    _print_guardrail_report("injection_guardrail_rate", injection_report)

    _write_results_report(golden_report, out_of_scope_report, injection_report)

    scores = _build_scores(golden_report, out_of_scope_report, injection_report)

    if args.mode == "gate":
        _run_gate(scores)
    elif args.mode == "update-baseline":
        _update_baseline(scores)


def _print_golden_report(report: GoldenReport) -> None:
    print("Golden Questions")
    if not report.has_data:
        print("  faithfulness: no data")
        print("  context_recall: no data")
        return

    total = len(report.scores)
    for score in report.scores:
        print(
            f"  - {score.question!r}: "
            f"faithfulness={_format_score(score.faithfulness)} "
            f"context_recall={_format_score(score.context_recall)}"
        )
    print(
        f"  faithfulness (mean): {_format_score(report.mean_faithfulness)} "
        f"({report.faithfulness_scored}/{total} scored)"
    )
    print(
        f"  context_recall (mean): {_format_score(report.mean_context_recall)} "
        f"({report.context_recall_scored}/{total} scored)"
    )


def _format_score(value: float | None) -> str:
    """`value` is `None` when the judge failed to score that question/metric (see
    `src.evaluation.golden._safe_judge_score`) -- printed as "ERROR", not 0.00."""
    return f"{value:.2f}" if value is not None else "ERROR"


def _print_guardrail_report(name: str, report: GuardrailReport) -> None:
    if not report.has_data:
        print(f"{name}: no data")
        return
    print(f"{name}: {report.rate:.2f} ({report.passed}/{report.total})")


def _write_results_report(
    golden_report: GoldenReport,
    out_of_scope_report: GuardrailReport,
    injection_report: GuardrailReport,
) -> None:
    markdown = render_markdown_report(
        golden_report, out_of_scope_report, injection_report, generated_at=datetime.now()
    )
    Path(RESULTS_REPORT_PATH).write_text(markdown, encoding="utf-8")
    print()
    print(f"Wrote {RESULTS_REPORT_PATH}")


def _build_scores(
    golden_report: GoldenReport,
    out_of_scope_report: GuardrailReport,
    injection_report: GuardrailReport,
) -> Scores:
    """Aggregate the three reports into one scores dict, matching `eval/baseline.json`'s shape."""
    return {
        "faithfulness": golden_report.mean_faithfulness,
        "context_recall": golden_report.mean_context_recall,
        "out_of_scope_guardrail_rate": out_of_scope_report.rate,
        "injection_guardrail_rate": injection_report.rate,
    }


def _run_gate(scores: Scores) -> None:
    baseline = _load_baseline()
    result = evaluate_gate(scores, baseline)

    print()
    if not result.passed:
        print(f"Gate: FAIL ({', '.join(result.regressed_metrics)})")
        sys.exit(1)
    print("Gate: PASS")


def _update_baseline(scores: Scores) -> None:
    """Overwrite `eval/baseline.json`, keeping each metric's previous value if `scores`
    reports "no data" for it (e.g. a question set that's transiently empty) — a run with
    no data for a metric must never regress a previously-established baseline value to
    `null`, which would silently stop the gate from checking it."""
    baseline = _load_baseline()
    merged: Scores = {
        "faithfulness": scores["faithfulness"] if scores["faithfulness"] is not None else baseline["faithfulness"],
        "context_recall": scores["context_recall"]
        if scores["context_recall"] is not None
        else baseline["context_recall"],
        "out_of_scope_guardrail_rate": scores["out_of_scope_guardrail_rate"]
        if scores["out_of_scope_guardrail_rate"] is not None
        else baseline["out_of_scope_guardrail_rate"],
        "injection_guardrail_rate": scores["injection_guardrail_rate"]
        if scores["injection_guardrail_rate"] is not None
        else baseline["injection_guardrail_rate"],
    }

    Path(BASELINE_PATH).write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"Updated {BASELINE_PATH}")


def _load_baseline() -> Scores:
    data = json.loads(Path(BASELINE_PATH).read_text(encoding="utf-8"))
    return Scores(
        faithfulness=data["faithfulness"],
        context_recall=data["context_recall"],
        out_of_scope_guardrail_rate=data["out_of_scope_guardrail_rate"],
        injection_guardrail_rate=data["injection_guardrail_rate"],
    )


if __name__ == "__main__":
    main()
