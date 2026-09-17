"""`scripts/evaluate.py` must be safe to run before the eval/ files are populated."""

import runpy
import sys

import pytest


def test_running_against_the_empty_seed_files_reports_no_data(capsys: pytest.CaptureFixture[str]) -> None:
    argv = sys.argv
    sys.argv = ["scripts/evaluate.py"]
    try:
        runpy.run_path("scripts/evaluate.py", run_name="__main__")
    finally:
        sys.argv = argv

    output = capsys.readouterr().out

    assert "faithfulness: no data" in output
    assert "context_recall: no data" in output
    assert "out_of_scope_guardrail_rate: no data" in output
    assert "injection_guardrail_rate: no data" in output
