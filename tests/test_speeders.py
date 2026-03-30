"""Tests for speeder detection."""

import pandas as pd
import pytest

from survey_cleaner.detectors.speeders import detect_speeders, _find_timing_column
from survey_cleaner.models import FlagCode


def make_df(**kwargs):
    return pd.DataFrame(kwargs)


# --- _find_timing_column ---


def test_find_timing_column_duration_seconds():
    df = make_df(duration_seconds=[100, 200], q1=[1, 2])
    assert _find_timing_column(df) == "duration_seconds"


def test_find_timing_column_case_insensitive():
    df = make_df(**{"Duration (in seconds)": [100, 200]})
    assert _find_timing_column(df) == "Duration (in seconds)"


def test_find_timing_column_missing():
    df = make_df(q1=[1, 2], q2=[3, 4])
    assert _find_timing_column(df) is None


# --- detect_speeders ---


def test_speeders_flags_below_threshold():
    df = make_df(duration_seconds=[30, 200, 90, 300], respondent_id=["A", "B", "C", "D"])
    flags = detect_speeders(df, min_seconds=120, id_col="respondent_id")
    assert len(flags) == 2
    indices = {f.row_index for f in flags}
    assert 0 in indices  # 30s
    assert 2 in indices  # 90s


def test_speeders_none_flagged_when_all_pass():
    df = make_df(duration_seconds=[200, 300, 150])
    flags = detect_speeders(df, min_seconds=120)
    assert flags == []


def test_speeders_returns_correct_flag_code():
    df = make_df(duration_seconds=[10])
    flags = detect_speeders(df, min_seconds=120)
    assert flags[0].reasons[0].code == FlagCode.SPEEDER


def test_speeders_severity_scales_with_ratio():
    df = make_df(duration_seconds=[60, 12])
    flags = detect_speeders(df, min_seconds=120)
    sev_60 = next(f for f in flags if f.row_index == 0).reasons[0].severity
    sev_12 = next(f for f in flags if f.row_index == 1).reasons[0].severity
    assert sev_12 > sev_60  # shorter → higher severity


def test_speeders_missing_timing_col_returns_empty():
    df = make_df(q1=[1, 2, 3])
    flags = detect_speeders(df, min_seconds=120)
    assert flags == []


def test_speeders_explicit_timing_col():
    df = make_df(my_time_col=[30, 300], q1=[1, 2])
    flags = detect_speeders(df, timing_col="my_time_col", min_seconds=120)
    assert len(flags) == 1 and flags[0].row_index == 0


def test_speeders_respondent_id_propagated():
    df = make_df(duration_seconds=[30], rid=["XYZ-001"])
    flags = detect_speeders(df, min_seconds=120, id_col="rid")
    assert flags[0].respondent_id == "XYZ-001"
