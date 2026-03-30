"""Detector modules for survey-cleaner."""

from survey_cleaner.detectors.openends import detect_gibberish
from survey_cleaner.detectors.speeders import detect_speeders
from survey_cleaner.detectors.straightliners import detect_straight_liners

__all__ = ["detect_speeders", "detect_straight_liners", "detect_gibberish"]
