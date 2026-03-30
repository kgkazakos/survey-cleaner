"""CLI tests using Click's test runner."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from survey_cleaner.cli import cli


FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_survey.csv"


def test_cli_clean_runs_successfully():
    runner = CliRunner()
    result = runner.invoke(cli, ["clean", str(FIXTURE_CSV), "--quiet"])
    # Exit code 0 or 1 (1 = flag rate > 20%, both are valid)
    assert result.exit_code in (0, 1)


def test_cli_clean_writes_output(tmp_path):
    runner = CliRunner()
    out = tmp_path / "cleaned.csv"
    result = runner.invoke(cli, ["clean", str(FIXTURE_CSV), "--output", str(out), "--quiet"])
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) > 0


def test_cli_clean_writes_json_report(tmp_path):
    runner = CliRunner()
    report_path = tmp_path / "report.json"
    result = runner.invoke(
        cli,
        ["clean", str(FIXTURE_CSV), "--report", str(report_path), "--quiet"],
    )
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert "total_responses" in data
    assert "flagged" in data


def test_cli_report_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["report", str(FIXTURE_CSV)])
    assert result.exit_code == 0


def test_cli_dry_run_no_files_written(tmp_path):
    runner = CliRunner()
    out = tmp_path / "cleaned.csv"
    runner.invoke(cli, ["clean", str(FIXTURE_CSV), "--output", str(out), "--dry-run", "--quiet"])
    assert not out.exists()
