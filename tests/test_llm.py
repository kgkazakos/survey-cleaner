"""Tests for LLM enricher — all backends mocked, no real API calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from survey_cleaner.llm import LLMEnricher, _parse_llm_response, _build_user_prompt


# --- _parse_llm_response ---


def test_parse_valid_json():
    raw = '[{"score": 0.9, "reason": "keyboard mash"}, {"score": 0.1, "reason": "coherent"}]'
    result = _parse_llm_response(raw, 2)
    assert len(result) == 2
    assert result[0]["score"] == 0.9
    assert result[1]["score"] == 0.1


def test_parse_strips_markdown_fences():
    raw = '```json\n[{"score": 0.5, "reason": "vague"}]\n```'
    result = _parse_llm_response(raw, 1)
    assert result[0]["score"] == 0.5


def test_parse_pads_short_response():
    raw = '[{"score": 0.8, "reason": "gibberish"}]'
    result = _parse_llm_response(raw, 3)
    assert len(result) == 3
    assert result[1]["score"] == 0.0  # padded


def test_parse_truncates_long_response():
    raw = '[{"score": 0.1, "reason": "ok"}, {"score": 0.9, "reason": "bad"}, {"score": 0.5, "reason": "meh"}]'
    result = _parse_llm_response(raw, 2)
    assert len(result) == 2


def test_parse_malformed_returns_fallback():
    result = _parse_llm_response("not json at all {{{{", 2)
    assert len(result) == 2
    assert all(r["score"] == 0.0 for r in result)


# --- _build_user_prompt ---


def test_build_user_prompt_contains_texts():
    texts = ["This is great", "asdfghjkl"]
    prompt = _build_user_prompt(texts)
    assert "This is great" in prompt
    assert "asdfghjkl" in prompt
    assert "2" in prompt


# --- LLMEnricher init ---


def test_enricher_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        LLMEnricher(backend="cohere")


def test_enricher_uses_default_model():
    e = LLMEnricher(backend="openai")
    assert "gpt" in e.model

    e2 = LLMEnricher(backend="anthropic")
    assert "claude" in e2.model

    e3 = LLMEnricher(backend="gemini")
    assert "gemini" in e3.model


def test_enricher_accepts_custom_model():
    e = LLMEnricher(backend="openai", model="gpt-4o")
    assert e.model == "gpt-4o"


# --- OpenAI backend (mocked) ---


def _make_openai_response(scores: list[tuple[float, str]]):
    """Build a mock openai ChatCompletion response."""
    import json

    content = json.dumps([{"score": s, "reason": r} for s, r in scores])
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def test_openai_backend_scores_responses():
    enricher = LLMEnricher(backend="openai")
    mock_response = _make_openai_response([(0.9, "keyboard mash"), (0.05, "genuine feedback")])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch.object(enricher, "_get_openai_client", return_value=mock_client):
        results = enricher.score_responses(["asdfghjkl", "Great tool, saves time every day"])

    assert len(results) == 2
    assert results[0][0] == pytest.approx(0.9)
    assert results[1][0] == pytest.approx(0.05)


def test_openai_backend_missing_import():
    enricher = LLMEnricher(backend="openai")
    with patch.dict("sys.modules", {"openai": None}):
        with pytest.raises(ImportError, match="OpenAI backend"):
            enricher._get_openai_client()


# --- Anthropic backend (mocked) ---


def _make_anthropic_response(scores: list[tuple[float, str]]):
    import json

    content_block = MagicMock()
    content_block.text = json.dumps([{"score": s, "reason": r} for s, r in scores])
    response = MagicMock()
    response.content = [content_block]
    return response


def test_anthropic_backend_scores_responses():
    enricher = LLMEnricher(backend="anthropic")
    mock_response = _make_anthropic_response([(0.85, "repeated single word"), (0.0, "clear and relevant")])

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch.object(enricher, "_get_anthropic_client", return_value=mock_client):
        results = enricher.score_responses(["no no no no", "The dashboard is intuitive and well-designed"])

    assert len(results) == 2
    assert results[0][0] == pytest.approx(0.85)
    assert results[1][0] == pytest.approx(0.0)


def test_anthropic_backend_missing_import():
    enricher = LLMEnricher(backend="anthropic")
    with patch.dict("sys.modules", {"anthropic": None}):
        with pytest.raises(ImportError, match="Anthropic backend"):
            enricher._get_anthropic_client()


# --- Gemini backend (mocked) ---


def _make_gemini_response(scores: list[tuple[float, str]]):
    import json

    response = MagicMock()
    response.text = json.dumps([{"score": s, "reason": r} for s, r in scores])
    return response


def test_gemini_backend_scores_responses():
    enricher = LLMEnricher(backend="gemini")
    mock_response = _make_gemini_response([(0.0, "genuine"), (1.0, "gibberish")])

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch.object(enricher, "_get_gemini_client", return_value=mock_model):
        results = enricher.score_responses(["I appreciated the fast support response", "qwerty asdf zxcv"])

    assert len(results) == 2
    assert results[0][0] == pytest.approx(0.0)
    assert results[1][0] == pytest.approx(1.0)


def test_gemini_backend_missing_import():
    enricher = LLMEnricher(backend="gemini")
    with patch.dict("sys.modules", {"google.generativeai": None, "google": None}):
        with pytest.raises((ImportError, AttributeError)):
            enricher._get_gemini_client()


# --- Batching ---


def test_batching_splits_into_multiple_calls():
    enricher = LLMEnricher(backend="openai", batch_size=2)
    texts = ["text A", "text B", "text C", "text D", "text E"]

    call_count = 0

    def fake_score_batch(batch):
        nonlocal call_count
        call_count += 1
        return [{"score": 0.1, "reason": "ok"}] * len(batch)

    with patch.object(enricher, "_score_batch", side_effect=fake_score_batch):
        results = enricher.score_responses(texts)

    assert len(results) == 5
    assert call_count == 3  # ceil(5/2)


# --- Error resilience ---


def test_enricher_returns_zeros_on_backend_error():
    enricher = LLMEnricher(backend="openai")

    def raise_error(batch):
        raise ConnectionError("API unreachable")

    with patch.object(enricher, "_score_batch", side_effect=raise_error):
        results = enricher.score_responses(["some text", "other text"])

    assert len(results) == 2
    assert all(score == 0.0 for score, _ in results)


# --- Integration: detect_gibberish with mocked LLM ---


def test_detect_gibberish_with_llm_enrichment():
    """LLM catches a response that the rule-based scorer missed."""
    from survey_cleaner.detectors.openends import detect_gibberish

    # "I am a user" is long enough and has decent entropy — rule-based won't flag it
    # But a mocked LLM considers it off-topic (score 0.8)
    df = pd.DataFrame({"open_comment": ["I am a user", "Excellent tool for our team's workflow"]})

    mock_enricher = MagicMock()
    mock_enricher.score_responses.return_value = [
        (0.8, "Off-topic non-answer"),
        (0.0, "Genuine and relevant"),
    ]

    with patch("survey_cleaner.llm.LLMEnricher", return_value=mock_enricher):
        flags = detect_gibberish(
            df,
            open_end_cols=["open_comment"],
            llm_backend="openai",
        )

    assert len(flags) == 1
    assert flags[0].row_index == 0
    assert "llm" in flags[0].reasons[0].detail


def test_detect_gibberish_llm_merged_with_rule_score():
    """Final score is max(rule_score, llm_score)."""
    from survey_cleaner.detectors.openends import detect_gibberish

    df = pd.DataFrame({"open_comment": ["asdfghjkl qwerty"]})  # rule-based scores high

    mock_enricher = MagicMock()
    mock_enricher.score_responses.return_value = [(0.3, "borderline")]  # LLM scores lower

    with patch("survey_cleaner.llm.LLMEnricher", return_value=mock_enricher):
        flags = detect_gibberish(
            df,
            open_end_cols=["open_comment"],
            llm_backend="anthropic",
        )

    # Should still be flagged because rule-based score was high
    assert len(flags) == 1
