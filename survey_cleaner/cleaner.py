"""Main SurveyCleaner orchestrator."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from survey_cleaner.detectors.openends import detect_gibberish
from survey_cleaner.detectors.speeders import detect_speeders
from survey_cleaner.detectors.straightliners import detect_straight_liners
from survey_cleaner.models import CleaningResult, FlagCode, FlaggedResponse


class SurveyCleaner:
    """
    Orchestrates all survey quality detectors and produces a CleaningResult.

    Parameters
    ----------
    timing_col:
        Column containing completion time in seconds. Auto-detected if None.
    min_seconds:
        Minimum acceptable completion time (default 120 s).
    likert_cols:
        Explicit Likert columns. Auto-detected if None.
    variance_threshold:
        Straight-liner variance threshold (default 0.0 = pure straight-liners).
    check_patterns:
        Whether to detect repeating Likert patterns.
    open_end_cols:
        Explicit open-end text columns. Auto-detected if None.
    min_chars:
        Minimum acceptable open-end length.
    gibberish_score_threshold:
        Composite gibberish score cutoff (0–1).
    id_col:
        Respondent identifier column name.
    run_speeders:
        Toggle speeder detection.
    run_straight_liners:
        Toggle straight-liner / pattern-liner detection.
    run_gibberish:
        Toggle open-end gibberish detection.
    llm_backend:
        Optional LLM backend for open-end enrichment (openai / anthropic / gemini).
    """

    def __init__(
        self,
        timing_col: Optional[str] = None,
        min_seconds: int = 120,
        likert_cols: Optional[list[str]] = None,
        variance_threshold: float = 0.0,
        check_patterns: bool = True,
        open_end_cols: Optional[list[str]] = None,
        min_chars: int = 10,
        gibberish_score_threshold: float = 0.5,
        id_col: Optional[str] = None,
        run_speeders: bool = True,
        run_straight_liners: bool = True,
        run_gibberish: bool = True,
        llm_backend: Optional[str] = None,
    ) -> None:
        self.timing_col = timing_col
        self.min_seconds = min_seconds
        self.likert_cols = likert_cols
        self.variance_threshold = variance_threshold
        self.check_patterns = check_patterns
        self.open_end_cols = open_end_cols
        self.min_chars = min_chars
        self.gibberish_score_threshold = gibberish_score_threshold
        self.id_col = id_col
        self.run_speeders = run_speeders
        self.run_straight_liners = run_straight_liners
        self.run_gibberish = run_gibberish
        self.llm_backend = llm_backend

    def clean(self, df: pd.DataFrame) -> CleaningResult:
        """
        Run all enabled detectors and return a CleaningResult.

        The returned result contains:
        - which rows are flagged and why
        - the clean row indices
        - per-detector counts
        """
        # Merge flags by row index
        flags_by_idx: dict[int, FlaggedResponse] = {}

        def _merge(new_flags: list[FlaggedResponse]) -> None:
            for fr in new_flags:
                if fr.row_index in flags_by_idx:
                    flags_by_idx[fr.row_index].reasons.extend(fr.reasons)
                else:
                    flags_by_idx[fr.row_index] = fr

        speeder_indices: set[int] = set()
        sl_indices: set[int] = set()
        pl_indices: set[int] = set()
        gibberish_indices: set[int] = set()

        if self.run_speeders:
            speeder_flags = detect_speeders(
                df,
                timing_col=self.timing_col,
                min_seconds=self.min_seconds,
                id_col=self.id_col,
            )
            speeder_indices = {f.row_index for f in speeder_flags}
            _merge(speeder_flags)

        if self.run_straight_liners:
            sl_flags = detect_straight_liners(
                df,
                likert_cols=self.likert_cols,
                variance_threshold=self.variance_threshold,
                check_patterns=self.check_patterns,
                id_col=self.id_col,
                exclude_cols=[self.id_col] if self.id_col else None,
            )
            for f in sl_flags:
                for r in f.reasons:
                    if r.code == FlagCode.STRAIGHT_LINER:
                        sl_indices.add(f.row_index)
                    elif r.code == FlagCode.PATTERN_LINER:
                        pl_indices.add(f.row_index)
            _merge(sl_flags)

        if self.run_gibberish:
            gibberish_flags = detect_gibberish(
                df,
                open_end_cols=self.open_end_cols,
                min_chars=self.min_chars,
                gibberish_score_threshold=self.gibberish_score_threshold,
                id_col=self.id_col,
                exclude_cols=[self.id_col] if self.id_col else None,
                llm_backend=self.llm_backend,
            )
            gibberish_indices = {f.row_index for f in gibberish_flags}
            _merge(gibberish_flags)

        all_flagged_indices = set(flags_by_idx.keys())
        all_indices = set(df.index.tolist())
        clean_indices = sorted(all_indices - all_flagged_indices)

        return CleaningResult(
            total_responses=len(df),
            flagged=list(flags_by_idx.values()),
            clean_indices=clean_indices,
            speeder_count=len(speeder_indices),
            straight_liner_count=len(sl_indices),
            pattern_liner_count=len(pl_indices),
            gibberish_count=len(gibberish_indices),
            config={
                "min_seconds": self.min_seconds,
                "variance_threshold": self.variance_threshold,
                "min_chars": self.min_chars,
                "gibberish_score_threshold": self.gibberish_score_threshold,
                "run_speeders": self.run_speeders,
                "run_straight_liners": self.run_straight_liners,
                "run_gibberish": self.run_gibberish,
            },
        )

    def clean_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningResult]:
        """
        Convenience method: returns (cleaned_df, result).

        The cleaned DataFrame contains only rows that passed all detectors,
        with the original index preserved.
        """
        result = self.clean(df)
        cleaned = df.loc[result.clean_indices].copy()
        return cleaned, result
