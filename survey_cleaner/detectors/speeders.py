"""Speeder detection: flags respondents who completed the survey too quickly."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from survey_cleaner.models import FlagCode, FlaggedResponse, FlagReason

# Common column names used by survey platforms for completion time
TIMING_COLUMN_ALIASES = {
    "duration",
    "duration_seconds",
    "duration (in seconds)",
    "time_taken",
    "time_taken_seconds",
    "timing",
    "interview_time",
    "response_time",
    "elapsed_time",
    "completion_time",
    "totalduration",
}

DEFAULT_MIN_SECONDS = 120


def _find_timing_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the timing column from common platform names."""
    lower_map = {col.lower().strip(): col for col in df.columns}
    for alias in TIMING_COLUMN_ALIASES:
        if alias in lower_map:
            return lower_map[alias]
    return None


def detect_speeders(
    df: pd.DataFrame,
    timing_col: Optional[str] = None,
    min_seconds: int = DEFAULT_MIN_SECONDS,
    id_col: Optional[str] = None,
) -> list[FlaggedResponse]:
    """
    Flag respondents whose completion time is below ``min_seconds``.

    Parameters
    ----------
    df:
        Survey DataFrame (one row per respondent).
    timing_col:
        Column containing completion time in seconds. Auto-detected if None.
    min_seconds:
        Minimum acceptable completion time. Rows below this are flagged.
    id_col:
        Column to use as respondent identifier in the output.

    Returns
    -------
    List of FlaggedResponse objects for rows that are speeders.
    """
    col = timing_col or _find_timing_column(df)
    if col is None:
        return []

    if col not in df.columns:
        raise ValueError(f"Timing column '{col}' not found in DataFrame.")

    times = pd.to_numeric(df[col], errors="coerce")
    flagged: list[FlaggedResponse] = []

    for idx, seconds in times.items():
        if pd.isna(seconds) or seconds >= min_seconds:
            continue

        ratio = float(seconds) / min_seconds  # 0 → 1 (lower = worse)
        severity = max(0.0, 1.0 - ratio)

        respondent_id = str(df.at[idx, id_col]) if id_col and id_col in df.columns else None

        flagged.append(
            FlaggedResponse(
                row_index=int(idx),
                respondent_id=respondent_id,
                reasons=[
                    FlagReason(
                        code=FlagCode.SPEEDER,
                        detail=f"Completed in {int(seconds)}s (minimum: {min_seconds}s)",
                        severity=round(severity, 3),
                    )
                ],
            )
        )

    return flagged
