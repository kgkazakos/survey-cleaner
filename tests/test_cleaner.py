"""Integration tests for SurveyCleaner."""

import pandas as pd
import pytest

from survey_cleaner.cleaner import SurveyCleaner
from survey_cleaner.models import FlagCode


LIKERT_COLS = ["q1", "q2", "q3", "q4", "q5"]
OPEN_END_COLS = ["open_feedback"]


def make_clean_df(n=5):
    """Produce a DataFrame with no quality issues."""
    return pd.DataFrame(
        {
            "duration_seconds": [300, 250, 400, 350, 200],
            "q1": [4, 3, 5, 2, 4],
            "q2": [3, 4, 2, 5, 3],
            "q3": [5, 2, 4, 3, 4],
            "q4": [2, 5, 3, 4, 2],
            "q5": [4, 3, 4, 2, 5],
            "open_feedback": [
                "Very helpful product for my team's workflow.",
                "Could use better documentation on the API.",
                "Dashboard is excellent but onboarding needs work.",
                "The mobile app needs improvement.",
                "Really satisfied with the support team.",
            ],
        }
    )


def make_problem_df():
    return pd.DataFrame(
        {
            "ResponseId": ["R01", "R02", "R03", "R04", "R05"],
            "duration_seconds": [300, 30, 250, 400, 80],        # R02, R05 → speeders
            "q1": [4, 3, 5, 5, 2],
            "q2": [3, 3, 5, 3, 4],
            "q3": [5, 3, 5, 4, 1],
            "q4": [2, 3, 5, 2, 3],
            "q5": [4, 3, 5, 5, 5],                               # R03 → straight-liner
            "open_feedback": [
                "Great product for our team.",
                "asdfghjkl qwerty mash",                         # R02 → gibberish
                "Amazing experience.",
                "Works well for our use case.",
                "ok",                                            # R05 → too short
            ],
        }
    )


# --- basic smoke tests ---


def test_clean_df_no_issues():
    df = make_clean_df()
    cleaner = SurveyCleaner(
        likert_cols=LIKERT_COLS,
        open_end_cols=OPEN_END_COLS,
    )
    cleaned, result = cleaner.clean_df(df)
    assert result.flagged_count == 0
    assert result.clean_count == 5
    assert len(cleaned) == 5


def test_clean_df_detects_all_issue_types():
    df = make_problem_df()
    cleaner = SurveyCleaner(
        likert_cols=LIKERT_COLS,
        open_end_cols=OPEN_END_COLS,
        id_col="ResponseId",
        min_seconds=120,
    )
    cleaned, result = cleaner.clean_df(df)
    assert result.speeder_count >= 1
    assert result.flagged_count >= 2


def test_clean_df_shape():
    df = make_problem_df()
    cleaner = SurveyCleaner(
        likert_cols=LIKERT_COLS,
        open_end_cols=OPEN_END_COLS,
        min_seconds=120,
    )
    cleaned, result = cleaner.clean_df(df)
    assert len(cleaned) == result.clean_count
    assert result.total_responses == len(df)


def test_clean_toggles_speeders_off():
    df = make_problem_df()
    cleaner = SurveyCleaner(
        likert_cols=LIKERT_COLS,
        open_end_cols=OPEN_END_COLS,
        min_seconds=120,
        run_speeders=False,
    )
    _, result = cleaner.clean_df(df)
    assert result.speeder_count == 0


def test_clean_toggles_gibberish_off():
    df = make_problem_df()
    cleaner = SurveyCleaner(
        likert_cols=LIKERT_COLS,
        open_end_cols=OPEN_END_COLS,
        run_gibberish=False,
    )
    _, result = cleaner.clean_df(df)
    assert result.gibberish_count == 0


def test_to_dict_is_serializable():
    import json
    df = make_problem_df()
    cleaner = SurveyCleaner(likert_cols=LIKERT_COLS, open_end_cols=OPEN_END_COLS)
    result = cleaner.clean(df)
    serialized = json.dumps(result.to_dict())
    parsed = json.loads(serialized)
    assert parsed["total_responses"] == len(df)


def test_flag_rate_calculation():
    df = make_problem_df()
    cleaner = SurveyCleaner(
        likert_cols=LIKERT_COLS,
        open_end_cols=OPEN_END_COLS,
        min_seconds=120,
    )
    result = cleaner.clean(df)
    expected = result.flagged_count / result.total_responses
    assert abs(result.flag_rate - expected) < 1e-6


def test_clean_indices_are_complement_of_flagged():
    df = make_problem_df()
    cleaner = SurveyCleaner(
        likert_cols=LIKERT_COLS,
        open_end_cols=OPEN_END_COLS,
        min_seconds=120,
    )
    result = cleaner.clean(df)
    all_idx = set(df.index.tolist())
    flagged_idx = {f.row_index for f in result.flagged}
    clean_idx = set(result.clean_indices)
    assert flagged_idx | clean_idx == all_idx
    assert flagged_idx & clean_idx == set()
