"""Straight-liner and pattern-liner detection for Likert-scale columns."""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from survey_cleaner.models import FlagCode, FlaggedResponse, FlagReason

# Typical Likert scale ranges
LIKERT_RANGES = [{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5, 6}]
LIKERT_MAX_UNIQUE = 7
LIKERT_MIN_COLS = 3  # need at least 3 rating columns to make a meaningful call

DEFAULT_VARIANCE_THRESHOLD = 0.0   # pure straight-liner: zero variance
DEFAULT_PATTERN_MAX_PERIOD = 3     # detect repeating patterns with period ≤ 3


def _auto_detect_likert_cols(df: pd.DataFrame, exclude: Optional[list[str]] = None) -> list[str]:
    """
    Heuristically identify Likert-scale columns.

    Criteria: numeric, integer-like values, ≤ LIKERT_MAX_UNIQUE unique values
    across the column, and the unique values fit a known Likert range.
    """
    exclude_set = set(exclude or [])
    likert_cols = []
    for col in df.columns:
        if col in exclude_set:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        if not pd.api.types.is_numeric_dtype(series):
            continue
        # Must be integer-like
        if not (series == series.round()).all():
            continue
        unique_vals = set(series.unique())
        if len(unique_vals) > LIKERT_MAX_UNIQUE:
            continue
        if any(unique_vals.issubset(lr) for lr in LIKERT_RANGES):
            likert_cols.append(col)
    return likert_cols


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _detect_repeating_pattern(values: list[int], max_period: int = DEFAULT_PATTERN_MAX_PERIOD) -> Optional[int]:
    """
    Return the detected period if the sequence is a strict repeating pattern,
    else None.  Requires at least 2 full cycles.
    """
    n = len(values)
    for period in range(1, max_period + 1):
        if n < period * 2:
            continue
        pattern = values[:period]
        full_cycles = n // period
        expected = (pattern * (full_cycles + 1))[:n]
        if values == expected:
            return period
    return None


def detect_straight_liners(
    df: pd.DataFrame,
    likert_cols: Optional[list[str]] = None,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    check_patterns: bool = True,
    pattern_max_period: int = DEFAULT_PATTERN_MAX_PERIOD,
    id_col: Optional[str] = None,
    exclude_cols: Optional[list[str]] = None,
) -> list[FlaggedResponse]:
    """
    Flag straight-liners (zero/near-zero variance across Likert columns)
    and pattern-liners (repeating response sequences).

    Parameters
    ----------
    df:
        Survey DataFrame.
    likert_cols:
        Explicitly specified Likert columns. Auto-detected if None.
    variance_threshold:
        Rows with variance ≤ this value are flagged as straight-liners.
        Default 0.0 catches only pure straight-liners.
    check_patterns:
        If True, also check for repeating pattern responses.
    pattern_max_period:
        Maximum repeating period to check (e.g. 3 catches 1,2,3,1,2,3...).
    id_col:
        Column to use as respondent identifier.
    exclude_cols:
        Columns to exclude from auto-detection.

    Returns
    -------
    List of FlaggedResponse objects.
    """
    cols = likert_cols or _auto_detect_likert_cols(df, exclude=exclude_cols)
    if len(cols) < LIKERT_MIN_COLS:
        return []

    flagged: list[FlaggedResponse] = []

    for idx, row in df.iterrows():
        values_raw = [row[c] for c in cols]
        values_num = [v for v in values_raw if not (isinstance(v, float) and math.isnan(v))]

        if len(values_num) < LIKERT_MIN_COLS:
            continue

        respondent_id = str(df.at[idx, id_col]) if id_col and id_col in df.columns else None
        reasons: list[FlagReason] = []

        # --- straight-liner check ---
        var = _variance([float(v) for v in values_num])
        if var <= variance_threshold:
            constant_val = values_num[0]
            severity = 1.0 if var == 0.0 else round(1.0 - (var / max(variance_threshold, 1e-9)), 3)
            reasons.append(
                FlagReason(
                    code=FlagCode.STRAIGHT_LINER,
                    detail=(
                        f"Constant rating '{constant_val}' across {len(values_num)} "
                        f"Likert items (variance={var:.4f})"
                    ),
                    severity=min(1.0, severity),
                )
            )

        # --- pattern-liner check (only if not already a straight-liner) ---
        if check_patterns and not reasons:
            int_vals = [int(v) for v in values_num]
            period = _detect_repeating_pattern(int_vals, max_period=pattern_max_period)
            if period is not None:
                pattern_str = ",".join(str(v) for v in int_vals[:period])
                reasons.append(
                    FlagReason(
                        code=FlagCode.PATTERN_LINER,
                        detail=(
                            f"Repeating pattern [{pattern_str}] "
                            f"across {len(values_num)} Likert items (period={period})"
                        ),
                        severity=0.75,
                    )
                )

        if reasons:
            flagged.append(
                FlaggedResponse(
                    row_index=int(idx),
                    respondent_id=respondent_id,
                    reasons=reasons,
                )
            )

    return flagged
