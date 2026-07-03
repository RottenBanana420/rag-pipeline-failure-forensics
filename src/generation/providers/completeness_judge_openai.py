"""OpenAI answer-completeness judge provider.

``openai`` is imported lazily inside ``__init__`` so this module can be
imported without the package being present. Tests should patch
``openai.OpenAI`` directly — Python's module cache means the patch is
visible to the inline import.

Uses the stable (non-beta) ``client.chat.completions.parse(...,
response_format=...)`` structured-output API — same call shape as
``OpenAICitationJudge`` (``citation_judge_openai.py``), re-confirmed via
Context7 against the currently installed ``openai`` SDK (v2.44.0) before
writing this file: ``.parse()`` accepts ``model``, ``messages``,
``response_format`` and ``temperature``, and returns a
``ParsedChatCompletion`` whose ``choices[0].message`` carries both
``parsed`` (the structured result, or ``None`` on refusal) and ``refusal``
fields. No drift found. Requires ``openai>=1.92.0`` (already the floor set
for the ``embed-openai`` extra by the citation judge feature).
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

DEFAULT_MODEL = "gpt-4o-2024-08-06"


class OpenAICompletenessJudge:
    """Answer completeness judge backed by the OpenAI Chat Completions structured output."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # lazy import — not at module level

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.answer_completeness_judge_model
        self._temperature = settings.answer_completeness_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"openai/gpt-4o-2024-08-06"``."""
        return f"openai/{self._model}"

    def judge(self, question: str, answer: str) -> CompletenessVerdict:
        """Decide whether *answer* addresses every part of *question*."""
        prompt = build_completeness_judge_prompt(question, answer)
        completion = self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": ANSWER_COMPLETENESS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt.user},
            ],
            temperature=self._temperature,
            response_format=CompletenessVerdict,
        )
        message = completion.choices[0].message
        parsed = message.parsed
        if parsed is None:
            raise RuntimeError(
                f"OpenAI structured output returned no parsed result "
                f"(model={self._model}, refusal={message.refusal!r})"
            )
        return parsed
