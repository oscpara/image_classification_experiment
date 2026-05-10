import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_script_help(script_path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script_path, "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_prediction_wrapper_scripts_support_direct_execution():
    for script_path in ("src/promote_predictions.py", "src/ingest_predictions.py"):
        result = _run_script_help(script_path)

        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_retrain_wrapper_supports_direct_execution_and_legacy_database_flags():
    result = _run_script_help("src/retrain_model.py")

    assert result.returncode == 0, result.stderr
    assert "--database-url" in result.stdout
    assert "--training-samples-sql" in result.stdout
