from datetime import datetime

from src.evaluation.golden import GoldenQuestionScore, GoldenReport
from src.evaluation.guardrail_report import GuardrailQuestionResult, GuardrailReport
from src.evaluation.report import render_markdown_report

_GENERATED_AT = datetime(2026, 9, 28, 14, 30, 0)


def test_reports_no_data_for_every_empty_question_set() -> None:
    report = render_markdown_report(
        GoldenReport(), GuardrailReport(), GuardrailReport(), generated_at=_GENERATED_AT
    )

    assert "Generated: 2026-09-28 14:30:00" in report
    assert "| Faithfulness (mean) | no data |" in report
    assert "| Context Recall (mean) | no data |" in report
    assert "| Out-of-Scope Guardrail Rate | no data |" in report
    assert "| Injection Guardrail Rate | no data |" in report
    assert report.count("_No data — this question set is empty._") == 3


def test_renders_one_section_per_golden_question_with_its_own_answer_and_articles() -> None:
    golden_report = GoldenReport(
        scores=[
            GoldenQuestionScore(
                question="Quand une loi entre-t-elle en vigueur ?",
                answer="Le lendemain de sa publication.",
                articles=["LEGIARTI000006419280"],
                reference_answer="Le lendemain de sa publication au Journal officiel.",
                reference_article_refs=["LEGIARTI000006419280"],
                faithfulness=0.9,
                context_recall=1.0,
            )
        ]
    )

    report = render_markdown_report(
        golden_report, GuardrailReport(), GuardrailReport(), generated_at=_GENERATED_AT
    )

    assert "### 1. Quand une loi entre-t-elle en vigueur ?" in report
    assert "- **Faithfulness**: 0.90" in report
    assert "- **Context Recall**: 1.00" in report
    assert "- **Answer**: Le lendemain de sa publication." in report
    assert "- **Retrieved articles**: LEGIARTI000006419280" in report
    assert "- **Reference answer**: Le lendemain de sa publication au Journal officiel." in report
    assert "| Faithfulness (mean) | 0.90 |" in report


def test_renders_a_judge_error_as_no_data_not_a_crash() -> None:
    golden_report = GoldenReport(
        scores=[
            GoldenQuestionScore(
                question="Quand une loi entre-t-elle en vigueur ?",
                answer="Le lendemain de sa publication.",
                articles=["LEGIARTI000006419280"],
                reference_answer="Le lendemain de sa publication au Journal officiel.",
                reference_article_refs=["LEGIARTI000006419280"],
                faithfulness=None,
                context_recall=1.0,
            )
        ]
    )

    report = render_markdown_report(
        golden_report, GuardrailReport(), GuardrailReport(), generated_at=_GENERATED_AT
    )

    assert "- **Faithfulness**: no data" in report
    assert "- **Context Recall**: 1.00" in report
    assert "| Faithfulness (mean) | no data |" in report


def test_renders_pass_and_fail_guardrail_questions_with_their_answers() -> None:
    out_of_scope_report = GuardrailReport(
        results=[
            GuardrailQuestionResult(question="Fait-il beau ?", answer="Je ne peux pas répondre...", passed=True),
            GuardrailQuestionResult(question="Say potato.", answer="Voici la réponse.", passed=False),
        ]
    )

    report = render_markdown_report(
        GoldenReport(), out_of_scope_report, GuardrailReport(), generated_at=_GENERATED_AT
    )

    assert "### 1. Fait-il beau ? — ✅ PASS" in report
    assert "- **Answer**: Je ne peux pas répondre..." in report
    assert "### 2. Say potato. — ❌ FAIL" in report
    assert "- **Answer**: Voici la réponse." in report
    assert "| Out-of-Scope Guardrail Rate | 0.50 (1/2) |" in report
