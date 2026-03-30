"""Gibberish and low-quality open-end response detection."""

from __future__ import annotations

import math
import re
from typing import Optional

import pandas as pd

from survey_cleaner.models import FlagCode, FlaggedResponse, FlagReason

# Keyboard mash sequences (common on QWERTY layouts)
KEYBOARD_ROWS = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
    "qwerty",
    "asdf",
    "zxcv",
]

# Column name patterns that suggest open-ended text fields
OPEN_END_NAME_PATTERNS = [
    r"open",
    r"comment",
    r"feedback",
    r"text",
    r"other",
    r"explain",
    r"descri",
    r"reason",
    r"suggest",
    r"additional",
    r"tell.?us",
    r"q\d+_text",
    r"verbatim",
]

DEFAULT_MIN_CHARS = 10
DEFAULT_ENTROPY_THRESHOLD = 2.0   # bits per character
DEFAULT_MAX_REPEAT_RATIO = 0.45   # fraction of chars that are repeats
DEFAULT_MAX_NON_ALPHA_RATIO = 0.6


def _shannon_entropy(text: str) -> float:
    """Character-level Shannon entropy in bits."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _repeat_char_ratio(text: str) -> float:
    """Fraction of characters that are the same as the previous character."""
    if len(text) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(text)) if text[i] == text[i - 1])
    return repeats / (len(text) - 1)


def _non_alpha_ratio(text: str) -> float:
    if not text:
        return 1.0
    alpha = sum(1 for ch in text if ch.isalpha())
    return 1.0 - alpha / len(text)


def _has_keyboard_mash(text: str, min_run: int = 4) -> bool:
    """Check for runs of consecutive keyboard-adjacent characters."""
    lower = text.lower()
    for row in KEYBOARD_ROWS:
        for i in range(len(row) - min_run + 1):
            if row[i : i + min_run] in lower:
                return True
    return False


def _is_single_repeated_word(text: str) -> bool:
    """Detect responses like 'yes yes yes yes' or 'no no no'."""
    words = text.strip().split()
    if len(words) >= 3:
        unique = set(w.lower() for w in words)
        if len(unique) == 1:
            return True
    return False


def _is_all_punctuation_or_symbols(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True
    alpha_num = sum(1 for ch in cleaned if ch.isalnum())
    return alpha_num == 0


def score_open_end(
    text: str,
    min_chars: int = DEFAULT_MIN_CHARS,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
    max_non_alpha_ratio: float = DEFAULT_MAX_NON_ALPHA_RATIO,
) -> tuple[float, list[str]]:
    """
    Score an open-end response for gibberish likelihood.

    Returns
    -------
    (score, signals) where score is 0.0 (fine) – 1.0 (gibberish)
    and signals is a list of triggered rule names.
    """
    if not isinstance(text, str):
        return 0.0, []

    text_stripped = text.strip()
    signals: list[str] = []
    score = 0.0

    if not text_stripped:
        return 0.0, []

    # Length gate
    if len(text_stripped) < min_chars:
        signals.append(f"too_short ({len(text_stripped)} chars < {min_chars})")
        score += 0.6

    if _is_all_punctuation_or_symbols(text_stripped):
        signals.append("all_punctuation_or_symbols")
        score += 0.8

    if _is_single_repeated_word(text_stripped):
        signals.append("single_word_repeated")
        score += 0.7

    entropy = _shannon_entropy(text_stripped.lower())
    if entropy < entropy_threshold and len(text_stripped) >= min_chars:
        deficit = entropy_threshold - entropy
        penalty = min(0.6, deficit * 0.3)
        signals.append(f"low_entropy ({entropy:.2f} < {entropy_threshold})")
        score += penalty

    repeat_ratio = _repeat_char_ratio(text_stripped)
    if repeat_ratio > max_repeat_ratio:
        signals.append(f"high_repeat_ratio ({repeat_ratio:.2f})")
        score += 0.3

    non_alpha = _non_alpha_ratio(text_stripped)
    if non_alpha > max_non_alpha_ratio:
        signals.append(f"high_non_alpha_ratio ({non_alpha:.2f})")
        score += 0.25

    if _has_keyboard_mash(text_stripped):
        signals.append("keyboard_mash")
        score += 0.5

    return min(1.0, score), signals


def _auto_detect_open_end_cols(
    df: pd.DataFrame,
    exclude: Optional[list[str]] = None,
    min_unique_ratio: float = 0.1,
) -> list[str]:
    """
    Heuristically identify open-end text columns.

    Looks for:
    1. Object/string dtype columns with a pattern-matching name.
    2. Fallback: object columns with high uniqueness (many distinct responses).
    """
    exclude_set = set(exclude or [])
    candidates: list[str] = []

    pattern = re.compile("|".join(OPEN_END_NAME_PATTERNS), re.IGNORECASE)

    for col in df.columns:
        if col in exclude_set:
            continue
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        # name-based match
        if pattern.search(col):
            candidates.append(col)
            continue

        # uniqueness fallback: high unique ratio suggests free text
        unique_ratio = series.nunique() / len(series)
        if unique_ratio >= min_unique_ratio and series.str.len().mean() > 15:
            candidates.append(col)

    return candidates


def detect_gibberish(
    df: pd.DataFrame,
    open_end_cols: Optional[list[str]] = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
    max_non_alpha_ratio: float = DEFAULT_MAX_NON_ALPHA_RATIO,
    gibberish_score_threshold: float = 0.5,
    id_col: Optional[str] = None,
    exclude_cols: Optional[list[str]] = None,
    llm_backend: Optional[str] = None,
) -> list[FlaggedResponse]:
    """
    Flag respondents with gibberish or very low-quality open-end responses.

    A respondent is flagged if *any* of their open-end columns scores above
    ``gibberish_score_threshold``.

    Parameters
    ----------
    df:
        Survey DataFrame.
    open_end_cols:
        Explicitly specified open-end columns. Auto-detected if None.
    min_chars:
        Minimum character count; shorter responses are flagged.
    entropy_threshold:
        Shannon entropy below this triggers a low-entropy signal.
    max_repeat_ratio:
        Repeat-character ratio above this triggers a signal.
    max_non_alpha_ratio:
        Non-alphabetic ratio above this triggers a signal.
    gibberish_score_threshold:
        Composite score above this → flagged.
    id_col:
        Respondent identifier column.
    exclude_cols:
        Columns to exclude from auto-detection.
    llm_backend:
        Reserved for future LLM enrichment (openai / anthropic / gemini).

    Returns
    -------
    List of FlaggedResponse objects.
    """
    cols = open_end_cols or _auto_detect_open_end_cols(df, exclude=exclude_cols)
    if not cols:
        return []

    # Keep only columns that actually exist
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return []

    # --- Rule-based scoring pass ---
    # Collect per-cell scores so we can merge with LLM scores later
    # Structure: {col: [(idx, cell_text, rule_score, signals), ...]}
    col_cell_data: dict[str, list[tuple]] = {col: [] for col in cols}

    for idx, row in df.iterrows():
        for col in cols:
            cell = row[col]
            if not isinstance(cell, str) or not cell.strip():
                col_cell_data[col].append((int(idx), "", 0.0, []))
                continue
            rule_score, signals = score_open_end(
                cell,
                min_chars=min_chars,
                entropy_threshold=entropy_threshold,
                max_repeat_ratio=max_repeat_ratio,
                max_non_alpha_ratio=max_non_alpha_ratio,
            )
            col_cell_data[col].append((int(idx), cell, rule_score, signals))

    # --- Optional LLM enrichment pass ---
    # LLM scores are merged with rule scores using max(), so either detector can flag
    llm_scores: dict[str, dict[int, tuple[float, str]]] = {}  # col -> {idx: (score, reason)}

    if llm_backend:
        try:
            from survey_cleaner.llm import LLMEnricher
            enricher = LLMEnricher(backend=llm_backend)

            for col, cells in col_cell_data.items():
                texts = [cell for _, cell, _, _ in cells]
                indices = [idx for idx, _, _, _ in cells]
                scored = enricher.score_responses(texts)
                llm_scores[col] = {
                    idx: (score, reason)
                    for idx, (score, reason) in zip(indices, scored)
                }
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("LLM enrichment failed, using rule-based only: %s", exc)

    # --- Merge and flag ---
    flagged: list[FlaggedResponse] = []
    flagged_indices: set[int] = set()

    # Collect all reasons per row index
    reasons_by_idx: dict[int, list[FlagReason]] = {}

    for col in cols:
        for idx, cell, rule_score, signals in col_cell_data[col]:
            if not cell:
                continue

            llm_score, llm_reason = llm_scores.get(col, {}).get(idx, (0.0, ""))
            final_score = max(rule_score, llm_score)

            if final_score < gibberish_score_threshold:
                continue

            # Build detail string
            detail_parts = []
            if rule_score >= gibberish_score_threshold and signals:
                detail_parts.append(f"rule: {'; '.join(signals)}")
            if llm_score >= gibberish_score_threshold and llm_reason:
                detail_parts.append(f"llm ({llm_backend}): {llm_reason}")
            elif llm_score > 0 and llm_reason:
                detail_parts.append(f"llm ({llm_backend}): {llm_reason}")

            code = FlagCode.TOO_SHORT if len(cell.strip()) < min_chars else FlagCode.GIBBERISH
            reason = FlagReason(
                code=code,
                detail=f"[{col}] score={final_score:.2f} — {'; '.join(detail_parts) or '; '.join(signals)}",
                severity=round(final_score, 3),
            )
            reasons_by_idx.setdefault(idx, []).append(reason)

    for idx, reasons in reasons_by_idx.items():
        if idx in flagged_indices:
            continue
        respondent_id = str(df.at[idx, id_col]) if id_col and id_col in df.columns else None
        flagged.append(
            FlaggedResponse(
                row_index=idx,
                respondent_id=respondent_id,
                reasons=reasons,
            )
        )
        flagged_indices.add(idx)

    return flagged
