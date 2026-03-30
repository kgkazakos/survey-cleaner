"""Tests for open-end gibberish detection."""

import pandas as pd
import pytest

from survey_cleaner.detectors.openends import (
    detect_gibberish,
    score_open_end,
    _shannon_entropy,
    _repeat_char_ratio,
    _has_keyboard_mash,
    _is_single_repeated_word,
)
from survey_cleaner.models import FlagCode


# --- low-level helpers ---


def test_shannon_entropy_uniform():
    # "abcd" — four distinct chars, equal freq
    e = _shannon_entropy("abcd")
    assert abs(e - 2.0) < 0.01


def test_shannon_entropy_single_char():
    assert _shannon_entropy("aaaa") == 0.0


def test_shannon_entropy_empty():
    assert _shannon_entropy("") == 0.0


def test_repeat_char_ratio_all_same():
    assert _repeat_char_ratio("aaaa") == 1.0


def test_repeat_char_ratio_alternating():
    # "ababab" — no consecutive repeats
    assert _repeat_char_ratio("ababab") == 0.0


def test_keyboard_mash_detected():
    assert _has_keyboard_mash("asdfghjkl") is True


def test_keyboard_mash_not_detected():
    assert _has_keyboard_mash("The product was helpful") is False


def test_single_word_repeated():
    assert _is_single_repeated_word("no no no no") is True


def test_single_word_not_repeated():
    assert _is_single_repeated_word("this is a real sentence") is False


# --- score_open_end ---


def test_score_gibberish_keyboard_mash():
    score, signals = score_open_end("qwerty asdf zxcv")
    assert score >= 0.5
    assert any("keyboard_mash" in s for s in signals)


def test_score_short_response():
    score, signals = score_open_end("ok", min_chars=10)
    assert score >= 0.5
    assert any("too_short" in s for s in signals)


def test_score_real_response():
    text = "The onboarding experience was seamless and the dashboard is very intuitive."
    score, signals = score_open_end(text)
    assert score < 0.5


def test_score_repeated_word():
    score, _ = score_open_end("yes yes yes yes yes")
    assert score >= 0.5


# --- detect_gibberish ---


def test_detect_gibberish_flags_keyboard_mash():
    df = pd.DataFrame(
        {
            "open_comment": [
                "asdfghjkl qwerty mash",
                "Great product, loved the workflow",
            ]
        }
    )
    flags = detect_gibberish(df, open_end_cols=["open_comment"])
    assert len(flags) == 1
    assert flags[0].row_index == 0


def test_detect_gibberish_flags_too_short():
    df = pd.DataFrame({"open_comment": ["ok", "Very helpful, saves time every day"]})
    flags = detect_gibberish(df, open_end_cols=["open_comment"], min_chars=10)
    assert len(flags) == 1
    assert flags[0].row_index == 0


def test_detect_gibberish_clean_responses():
    df = pd.DataFrame(
        {
            "open_comment": [
                "The integration took longer than expected but support was responsive.",
                "Really appreciate the new export feature.",
            ]
        }
    )
    flags = detect_gibberish(df, open_end_cols=["open_comment"])
    assert flags == []


def test_detect_gibberish_no_open_end_cols():
    df = pd.DataFrame({"q1": [1, 2, 3], "q2": [4, 5, 6]})
    flags = detect_gibberish(df)
    assert flags == []
