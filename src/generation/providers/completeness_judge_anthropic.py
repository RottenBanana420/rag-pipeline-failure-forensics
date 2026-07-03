"""Anthropic answer-completeness judge provider.

``anthropic`` is imported lazily inside ``__init__`` so this module can be
imported without the package being present. Tests should patch
``anthropic.Anthropic`` directly — Python's module cache means the patch is
visible to the inline import.

Uses the stable (non-beta) ``client.messages.parse(..., output_format=...)``
structured-output API — same call shape as ``AnthropicCitationJudge``
(``citation_judge_anthropic.py``), re-confirmed via Context7 against the
currently installed ``anthropic`` SDK (v0.116.0) before writing this file:
``client.messages.parse`` returns a ``ParsedMessage`` whose ``parsed_output``
property extracts the structured result validated against ``output_format``.
No drift from the pattern already used by ``AnthropicCitationJudge``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.generation.confidence_scorer import (
    ANSWER_COMPLETENESS_SYSTEM_PROMPT,
    CompletenessVerdict,
    build_completeness_judge_prompt,
)

if TYPE_CHECKING:
    from src.config import Settings

DEFAULT_MODEL = "claude-sonnet-4-5"


class AnthropicCompletenessJudge:
    """Answer completeness judge backed by the Anthropic Messages API structured output."""

    def __init__(self, settings: Settings) -> None:
        from anthropic import Anthropic  # lazy import — not at module level

        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.answer_completeness_judge_model
        self._temperature = settings.answer_completeness_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"anthropic/claude-sonnet-4-5"``."""
        return f"anthropic/{self._model}"

    def judge(self, question: str, answer: str) -> CompletenessVerdict:
        """Decide whether *answer* addresses every part of *question*."""
        prompt = build_completeness_judge_prompt(question, answer)
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=1024,
            system=ANSWER_COMPLETENESS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt.user}],
            temperature=self._temperature,
            output_format=CompletenessVerdict,
        )
        parsed = response.parsed_output
        if parsed is None:
            raise RuntimeError(
                f"Anthropic structured output returned no parsed_output "
                f"(model={self._model})"
            )
        return parsed
