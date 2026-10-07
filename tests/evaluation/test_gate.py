from src.evaluation.gate import Scores, evaluate_gate

_PASSING_BASELINE: Scores = {
    "faithfulness": 0.8,
    "context_recall": 0.7,
    "out_of_scope_guardrail_rate": 0.9,
    "injection_guardrail_rate": 1.0,
}

_NO_DATA_BASELINE: Scores = {
    "faithfulness": None,
    "context_recall": None,
    "out_of_scope_guardrail_rate": None,
    "injection_guardrail_rate": None,
}


def test_a_clean_pass_when_every_metric_meets_or_beats_the_baseline() -> None:
    scores: Scores = {
        "faithfulness": 0.8,
        "context_recall": 0.7,
        "out_of_scope_guardrail_rate": 0.9,
        "injection_guardrail_rate": 1.0,
    }

    result = evaluate_gate(scores, _PASSING_BASELINE)

    assert result.passed is True
    assert not result.regressed_metrics


def test_a_regression_on_faithfulness_alone_fails_the_gate() -> None:
    scores: Scores = {
        "faithfulness": 0.79,
        "context_recall": 0.7,
        "out_of_scope_guardrail_rate": 0.9,
        "injection_guardrail_rate": 1.0,
    }

    result = evaluate_gate(scores, _PASSING_BASELINE)

    assert result.passed is False
    assert result.regressed_metrics == ["faithfulness"]


def test_a_regression_on_context_recall_alone_fails_the_gate() -> None:
    scores: Scores = {
        "faithfulness": 0.8,
        "context_recall": 0.69,
        "out_of_scope_guardrail_rate": 0.9,
        "injection_guardrail_rate": 1.0,
    }

    result = evaluate_gate(scores, _PASSING_BASELINE)

    assert result.passed is False
    assert result.regressed_metrics == ["context_recall"]


def test_a_regression_on_out_of_scope_guardrail_rate_alone_fails_the_gate() -> None:
    scores: Scores = {
        "faithfulness": 0.8,
        "context_recall": 0.7,
        "out_of_scope_guardrail_rate": 0.89,
        "injection_guardrail_rate": 1.0,
    }

    result = evaluate_gate(scores, _PASSING_BASELINE)

    assert result.passed is False
    assert result.regressed_metrics == ["out_of_scope_guardrail_rate"]


def test_multiple_regressed_metrics_are_all_reported() -> None:
    scores: Scores = {
        "faithfulness": 0.5,
        "context_recall": 0.5,
        "out_of_scope_guardrail_rate": 0.9,
        "injection_guardrail_rate": 1.0,
    }

    result = evaluate_gate(scores, _PASSING_BASELINE)

    assert result.passed is False
    assert result.regressed_metrics == ["faithfulness", "context_recall"]


def test_an_improvement_on_one_metric_never_offsets_a_regression_on_another() -> None:
    scores: Scores = {
        "faithfulness": 1.0,
        "context_recall": 0.5,
        "out_of_scope_guardrail_rate": 0.9,
        "injection_guardrail_rate": 1.0,
    }

    result = evaluate_gate(scores, _PASSING_BASELINE)

    assert result.passed is False
    assert result.regressed_metrics == ["context_recall"]


def test_injection_guardrail_rate_below_1_0_fails_the_gate_even_above_its_baseline() -> None:
    baseline: Scores = {**_PASSING_BASELINE, "injection_guardrail_rate": 0.5}
    scores: Scores = {**_PASSING_BASELINE, "injection_guardrail_rate": 0.9}

    result = evaluate_gate(scores, baseline)

    assert result.passed is False
    assert result.regressed_metrics == ["injection_guardrail_rate"]


def test_injection_guardrail_rate_at_exactly_1_0_passes() -> None:
    result = evaluate_gate(_PASSING_BASELINE, _PASSING_BASELINE)

    assert result.passed is True
    assert not result.regressed_metrics


def test_a_metric_with_no_data_in_the_current_scores_is_skipped_not_failed() -> None:
    scores: Scores = {
        "faithfulness": None,
        "context_recall": None,
        "out_of_scope_guardrail_rate": None,
        "injection_guardrail_rate": None,
    }

    result = evaluate_gate(scores, _PASSING_BASELINE)

    assert result.passed is True
    assert not result.regressed_metrics


def test_the_no_data_bootstrap_case_passes_against_a_no_data_baseline() -> None:
    result = evaluate_gate(_NO_DATA_BASELINE, _NO_DATA_BASELINE)

    assert result.passed is True
    assert not result.regressed_metrics


def test_a_metric_with_data_but_no_baseline_yet_is_skipped_not_failed() -> None:
    scores: Scores = {
        "faithfulness": 0.1,
        "context_recall": 0.1,
        "out_of_scope_guardrail_rate": 0.1,
        "injection_guardrail_rate": 1.0,
    }

    result = evaluate_gate(scores, _NO_DATA_BASELINE)

    assert result.passed is True
    assert not result.regressed_metrics
