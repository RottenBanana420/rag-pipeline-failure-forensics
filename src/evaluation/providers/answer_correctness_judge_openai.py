"""OpenAI answer-correctness judge provider.

``openai`` is imported lazily inside ``__init__`` so that this module can be
imported without the package being present. Tests should patch
``openai.OpenAI`` directly — Python's module cache means the patch is
visible to the inline import.

Uses the stable (non-beta) ``client.chat.completions.parse(...,
response_format=...)`` structured-output API — same call shape as
``src.generation.providers.citation_judge_openai``, re-confirmed via
Context7 against the installed ``openai`` SDK (v2.44.0): ``.parse()`` accepts
``model``, ``messages``, ``response_format`` and ``temperature``, returning a
``ParsedChatCompletion`` whose ``choices[0].message`` carries ``parsed`` (or
``None`` on refusal) and ``refusal``.

Span step is ``"analysis"`` — see the Anthropic provider's docstring for why.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.evaluation.answer_correctness import (
    ANSWER_CORRECTNESS_JUDGE_SYSTEM_PROMPT,
    CorrectnessVerdict,
    build_answer_correctness_judge_prompt,
)
from src.tracing.instrumentation import default_serialize, span

if TYPE_CHECKING:
    from src.config import Settings

# Confirmed via Context7 (openai-python docs) and cross-checked against the
# installed openai SDK's ChatModel literal (v2.44.0): a current,
# structured-outputs-capable dated snapshot model.
DEFAULT_MODEL = "gpt-4o-2024-08-06"


def _extract_token_count(completion: object) -> int | None:
    usage = getattr(completion, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return total_tokens if isinstance(total_tokens, int) else None


class OpenAIAnswerCorrectnessJudge:
    """Answer-correctness judge backed by the OpenAI Chat Completions structured output."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # lazy import — not at module level

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.answer_correctness_judge_model
        self._temperature = settings.answer_correctness_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"openai/gpt-4o-2024-08-06"``."""
        return f"openai/{self._model}"

    def judge(
        self, question: str, expected_answer: str, actual_answer: str
    ) -> CorrectnessVerdict:
        """Decide whether *actual_answer* is correct relative to *expected_answer*."""
        prompt = build_answer_correctness_judge_prompt(
            question, expected_answer, actual_answer
        )
        with span(
            "analysis",
            input=default_serialize(
                {
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_answer": actual_answer,
                }
            ),
        ) as s:
            s.llm_prompt = f"{ANSWER_CORRECTNESS_JUDGE_SYSTEM_PROMPT}\n\n{prompt.user}"
            completion = self._client.chat.completions.parse(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": ANSWER_CORRECTNESS_JUDGE_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt.user},
                ],
                temperature=self._temperature,
                response_format=CorrectnessVerdict,
            )
            s.token_count = _extract_token_count(completion)
            message = completion.choices[0].message
            parsed = message.parsed
            if parsed is None:
                raise RuntimeError(
                    f"OpenAI structured output returned no parsed result "
                    f"(model={self._model}, refusal={message.refusal!r})"
                )
            s.output = default_serialize(parsed)
            return parsed
