"""Rendering a human-readable Markdown report from already-computed evaluation results.

Pure and side-effect-free: no I/O, no LLM, no network -- `scripts/evaluate.py`
does the actual `/query` calls and RAGAS scoring, then hands the resulting
reports here to be rendered. Kept separate so this formatting logic is unit
tested with hand-built fixtures, the same way `src.evaluation.gate` is.
"""

from __future__ import annotations

from datetime import datetime

from src.evaluation.golden import GoldenReport
from src.evaluation.guardrail_report import GuardrailReport


def render_markdown_report(
    golden_report: GoldenReport,
    out_of_scope_report: GuardrailReport,
    injection_report: GuardrailReport,
    *,
    generated_at: datetime,
) -> str:
    """Render a full Markdown report: summary table, then one section per question.

    Args:
        golden_report (GoldenReport): Golden Question scores, per-question and aggregate
        out_of_scope_report (GuardrailReport): Out-of-Scope Question pass/fail detail
        injection_report (GuardrailReport): Injection Attempt pass/fail detail
        generated_at (datetime): run timestamp, injected rather than read from the
            clock here so this function stays pure and deterministic to test

    Returns:
        str: the full report, as Markdown
    """
    sections = [
        "# RAG Evaluation Report",
        "",
        f"Generated: {generated_at:%Y-%m-%d %H:%M:%S}",
        "",
        _render_summary(golden_report, out_of_scope_report, injection_report),
        "",
        "## Golden Questions",
        "",
        _render_golden_questions(golden_report),
        "",
        "## Out-of-Scope Questions",
        "",
        _render_guardrail_questions(out_of_scope_report),
        "",
        "## Injection Attempts",
        "",
        _render_guardrail_questions(injection_report),
    ]
    return "\n".join(sections) + "\n"


def _render_summary(
    golden_report: GoldenReport,
    out_of_scope_report: GuardrailReport,
    injection_report: GuardrailReport,
) -> str:
    lines = [
        "## Summary",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        f"| Faithfulness (mean) | {_format_score(golden_report.mean_faithfulness)} |",
        f"| Context Recall (mean) | {_format_score(golden_report.mean_context_recall)} |",
        f"| Out-of-Scope Guardrail Rate | {_format_rate(out_of_scope_report)} |",
        f"| Injection Guardrail Rate | {_format_rate(injection_report)} |",
    ]
    return "\n".join(lines)


def _render_golden_questions(report: GoldenReport) -> str:
    if not report.has_data:
        return "_No data — this question set is empty._"

    blocks = []
    for i, score in enumerate(report.scores, start=1):
        blocks.append(
            "\n".join(
                [
                    f"### {i}. {score.question}",
                    "",
                    f"- **Faithfulness**: {_format_score(score.faithfulness)}",
                    f"- **Context Recall**: {_format_score(score.context_recall)}",
                    f"- **Answer**: {score.answer}",
                    f"- **Retrieved articles**: {_format_refs(score.articles)}",
                    f"- **Reference answer**: {score.reference_answer}",
                    f"- **Reference articles**: {_format_refs(score.reference_article_refs)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _render_guardrail_questions(report: GuardrailReport) -> str:
    if not report.has_data:
        return "_No data — this question set is empty._"

    blocks = []
    for i, result in enumerate(report.results, start=1):
        status = "✅ PASS" if result.passed else "❌ FAIL"
        blocks.append(
            "\n".join(
                [
                    f"### {i}. {result.question} — {status}",
                    "",
                    f"- **Answer**: {result.answer}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_score(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "no data"


def _format_rate(report: GuardrailReport) -> str:
    return f"{report.rate:.2f} ({report.passed}/{report.total})" if report.has_data else "no data"


def _format_refs(refs: list[str]) -> str:
    return ", ".join(refs) if refs else "(none)"
