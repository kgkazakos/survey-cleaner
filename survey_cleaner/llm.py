"""LLM enrichment for open-end response quality scoring.

Supports OpenAI, Anthropic, and Gemini backends. Install the relevant
optional extra to enable:

    pip install "survey-cleaner[openai]"
    pip install "survey-cleaner[anthropic]"
    pip install "survey-cleaner[gemini]"
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default models — cheapest/fastest option per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-3.1-pro-preview",
}

SYSTEM_PROMPT = """You are a survey data quality assessor. Your job is to evaluate open-end survey responses and identify low-quality submissions.

For each response provided, assign a quality score from 0.0 to 1.0:
- 0.0 = Genuine, thoughtful, coherent response — no quality issues
- 0.25 = Vague or minimal but plausibly genuine
- 0.5 = Borderline: off-topic, suspiciously generic, or too short to be meaningful
- 0.75 = Likely low-quality: irrelevant, filler, or clearly copy-pasted
- 1.0 = Definitely gibberish, keyboard mash, or meaningless noise

Return ONLY a JSON array with one object per response, in the same order as the input.
Each object must have exactly two keys: "score" (float 0.0–1.0) and "reason" (one short sentence).

Example output:
[
  {"score": 0.0, "reason": "Clear and relevant feedback about the product."},
  {"score": 1.0, "reason": "Keyboard mash with no meaningful content."}
]

Do not include any text outside the JSON array."""

USER_PROMPT_TEMPLATE = """Please score the following {n} open-end survey responses.

{items}

Return a JSON array of {n} objects."""


def _build_user_prompt(texts: list[str]) -> str:
    items = "\n".join(
        f"{i + 1}. {repr(text)}" for i, text in enumerate(texts)
    )
    return USER_PROMPT_TEMPLATE.format(n=len(texts), items=items)


def _parse_llm_response(raw: str, expected_count: int) -> list[dict]:
    """Parse LLM JSON response, with fallback for malformed output."""
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) == expected_count:
            return parsed
        logger.warning("LLM returned %d items, expected %d", len(parsed), expected_count)
        # Pad or truncate to match expected count
        while len(parsed) < expected_count:
            parsed.append({"score": 0.0, "reason": "No LLM score available."})
        return parsed[:expected_count]
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM response as JSON: %s", exc)
        return [{"score": 0.0, "reason": "LLM response parse error."}] * expected_count


class LLMEnricher:
    """
    Scores open-end survey responses using an LLM backend.

    Parameters
    ----------
    backend:
        One of 'openai', 'anthropic', 'gemini'.
    model:
        Model identifier. Defaults to the cheapest/fastest model per backend.
    api_key:
        Optional API key override. Reads from environment by default.
    batch_size:
        Maximum number of responses to score in a single API call.
    """

    def __init__(
        self,
        backend: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 20,
    ) -> None:
        if backend not in DEFAULT_MODELS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(DEFAULT_MODELS)}")
        self.backend = backend
        self.model = model or DEFAULT_MODELS[backend]
        self.api_key = api_key
        self.batch_size = batch_size
        self._client = None

    def _get_openai_client(self):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAI backend requires: pip install 'survey-cleaner[openai]'"
            ) from exc
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return openai.OpenAI(**kwargs)

    def _get_anthropic_client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic backend requires: pip install 'survey-cleaner[anthropic]'"
            ) from exc
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return anthropic.Anthropic(**kwargs)

    def _get_gemini_client(self):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "Gemini backend requires: pip install 'survey-cleaner[gemini]'"
            ) from exc
        if self.api_key:
            genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(
            model_name=self.model,
            system_instruction=SYSTEM_PROMPT,
        )

    def _score_batch_openai(self, texts: list[str]) -> list[dict]:
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(texts)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"} if "gpt-4" in self.model else None,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or "[]"
        # json_object mode wraps in an object sometimes — unwrap if needed
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                # Find the first list value
                for v in parsed.values():
                    if isinstance(v, list):
                        raw = json.dumps(v)
                        break
        except json.JSONDecodeError:
            pass
        return _parse_llm_response(raw, len(texts))

    def _score_batch_anthropic(self, texts: list[str]) -> list[dict]:
        client = self._get_anthropic_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _build_user_prompt(texts)},
            ],
            temperature=0.0,
        )
        raw = response.content[0].text if response.content else "[]"
        return _parse_llm_response(raw, len(texts))

    def _score_batch_gemini(self, texts: list[str]) -> list[dict]:
        client = self._get_gemini_client()
        response = client.generate_content(
            _build_user_prompt(texts),
            generation_config={"temperature": 0.0, "max_output_tokens": 1024},
        )
        raw = response.text if response.text else "[]"
        return _parse_llm_response(raw, len(texts))

    def _score_batch(self, texts: list[str]) -> list[dict]:
        """Score one batch via the configured backend."""
        if self.backend == "openai":
            return self._score_batch_openai(texts)
        elif self.backend == "anthropic":
            return self._score_batch_anthropic(texts)
        elif self.backend == "gemini":
            return self._score_batch_gemini(texts)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def score_responses(self, texts: list[str]) -> list[tuple[float, str]]:
        """
        Score a list of open-end text responses.

        Parameters
        ----------
        texts:
            List of raw response strings to evaluate.

        Returns
        -------
        List of (score, reason) tuples in the same order as ``texts``.
        Score is 0.0 (genuine) – 1.0 (gibberish).
        """
        if not texts:
            return []

        results: list[tuple[float, str]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                scored = self._score_batch(batch)
                for item in scored:
                    score = float(item.get("score", 0.0))
                    reason = str(item.get("reason", ""))
                    results.append((max(0.0, min(1.0, score)), reason))
            except Exception as exc:
                logger.error("LLM enrichment batch failed: %s", exc)
                results.extend([(0.0, f"LLM error: {exc}")] * len(batch))

        return results
