"""`scripts/evaluate.py` must be safe to run before the eval/ files are populated."""

import runpy
import sys
from pathlib import Path

import pytest

_EVALUATE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"


def test_running_against_the_empty_seed_files_reports_no_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Runs against genuinely empty JSONL files in an isolated cwd, not the repo's real
    `eval/` files -- those are populated with real Golden Questions (see CONTEXT.md's
    Evaluation section), which would otherwise pull in `ragas` and hit the real pipeline."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "golden_questions.jsonl").write_text("")
    (eval_dir / "out_of_scope_questions.jsonl").write_text("")
    (eval_dir / "injection_attempts.jsonl").write_text("")
    monkeypatch.chdir(tmp_path)

    argv = sys.argv
    sys.argv = ["scripts/evaluate.py"]
    try:
        runpy.run_path(str(_EVALUATE_SCRIPT), run_name="__main__")
    finally:
        sys.argv = argv

    output = capsys.readouterr().out

    assert "faithfulness: no data" in output
    assert "context_recall: no data" in output
    assert "out_of_scope_guardrail_rate: no data" in output
    assert "injection_guardrail_rate: no data" in output
