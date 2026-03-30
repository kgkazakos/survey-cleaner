"""Tests for straight-liner and pattern-liner detection."""

import pandas as pd
import pytest

from survey_cleaner.detectors.straightliners import (
    detect_straight_liners,
    _auto_detect_likert_cols,
    _variance,
    _detect_repeating_pattern,
)
from survey_cleaner.models import FlagCode


def make_df(**kwargs):
    return pd.DataFrame(kwargs)


# --- helpers ---


def test_variance_all_same():
    assert _variance([3.0, 3.0, 3.0]) == 0.0


def test_variance_mixed():
    assert _variance([1.0, 3.0, 5.0]) > 0


def test_detect_repeating_pattern_period_2():
    assert _detect_repeating_pattern([1, 2, 1, 2, 1, 2]) == 2


def test_detect_repeating_pattern_period_3():
    assert _detect_repeating_pattern([1, 2, 3, 1, 2, 3]) == 3


def test_detect_repeating_pattern_not_repeating():
    assert _detect_repeating_pattern([1, 2, 3, 4, 5]) is None


def test_detect_repeating_pattern_too_short_for_two_cycles():
    # Period 3 with only one cycle — not enough
    assert _detect_repeating_pattern([1, 2, 3]) is None


# --- auto-detect Likert cols ---


def test_auto_detect_likert_cols():
    df = make_df(
        q1=[1, 2, 3, 4, 5],
        q2=[5, 4, 3, 2, 1],
        q3=[3, 3, 4, 2, 5],
        text_col=["a", "b", "c", "d", "e"],
        big_num=[100, 200, 300, 400, 500],
    )
    cols = _auto_detect_likert_cols(df)
    assert set(cols) == {"q1", "q2", "q3"}


# --- detect_straight_liners ---


def test_straight_liner_pure():
    df = make_df(q1=[5, 3], q2=[5, 2], q3=[5, 4], q4=[5, 1], q5=[5, 3])
    flags = detect_straight_liners(df, likert_cols=["q1", "q2", "q3", "q4", "q5"])
    assert len(flags) == 1
    assert flags[0].row_index == 0
    assert flags[0].reasons[0].code == FlagCode.STRAIGHT_LINER


def test_straight_liner_not_flagged_when_varied():
    df = make_df(q1=[1, 3], q2=[5, 2], q3=[3, 4], q4=[2, 1], q5=[4, 5])
    flags = detect_straight_liners(df, likert_cols=["q1", "q2", "q3", "q4", "q5"])
    assert len(flags) == 0


def test_pattern_liner_detected():
    df = make_df(q1=[1, 3], q2=[2, 2], q3=[1, 4], q4=[2, 1], q5=[1, 5], q6=[2, 3])
    flags = detect_straight_liners(df, likert_cols=["q1", "q2", "q3", "q4", "q5", "q6"])
    pattern_flags = [f for f in flags if any(r.code == FlagCode.PATTERN_LINER for r in f.reasons)]
    assert any(f.row_index == 0 for f in pattern_flags)


def test_pattern_liner_not_flagged_when_disabled():
    df = make_df(q1=[1, 3], q2=[2, 2], q3=[1, 4], q4=[2, 1], q5=[1, 5], q6=[2, 3])
    flags = detect_straight_liners(
        df,
        likert_cols=["q1", "q2", "q3", "q4", "q5", "q6"],
        check_patterns=False,
    )
    pattern_flags = [f for f in flags if any(r.code == FlagCode.PATTERN_LINER for r in f.reasons)]
    assert len(pattern_flags) == 0


def test_too_few_likert_cols_returns_empty():
    df = make_df(q1=[3, 3], q2=[3, 2])
    flags = detect_straight_liners(df, likert_cols=["q1", "q2"])
    assert flags == []
