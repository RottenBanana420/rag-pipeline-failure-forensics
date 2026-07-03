"""OpenAI citation judge provider.

``openai`` is imported lazily inside ``__init__`` so that this module can be
imported without the package being present. Tests should patch
``openai.OpenAI`` directly — Python's module cache means the patch is
visible to the inline import.

Uses the stable (non-beta) ``client.chat.completions.parse(...,
response_format=...)`` structured-output API. Confirmed via Context7 against
the ``openai-python`` docs and verified against the installed ``openai`` SDK
(v2.44.0) source directly: ``.parse()`` accepts ``model``, ``messages``,
``response_format`` and ``temperature``, and returns a
``ParsedChatCompletion`` whose ``choices[0].message`` carries both
``parsed`` (the structured result, or ``None`` on refusal) and ``refusal``
fields. Requires ``openai>=1.92.0`` — see ``docs/DECISIONS.md`` for why the
version floor was bumped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.generation.citation_verifier import (
    CITATION_JUDGE_SYSTEM_PROMPT,
    JudgeVerdict,
    build_judge_prompt,
)

if TYPE_CHECKING:
    from src.config import Settings

# Confirmed via Context7 (openai-python docs, v2.11.0 snapshot) and cross-checked
# against the installed openai SDK's ChatModel literal (v2.44.0): a current,
# structured-outputs-capable dated snapshot model.
DEFAULT_MODEL = "gpt-4o-2024-08-06"


class OpenAICitationJudge:
    """Citation judge backed by the OpenAI Chat Completions structured output."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # lazy import — not at module level

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.citation_judge_model
        self._temperature = settings.citation_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"openai/gpt-4o-2024-08-06"``."""
        return f"openai/{self._model}"

    def judge(self, claim: str, evidence: str) -> JudgeVerdict:
        """Decide whether *evidence* supports *claim* and return a verdict."""
        prompt = build_judge_prompt(claim, evidence)
        completion = self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": CITATION_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt.user},
            ],
            temperature=self._temperature,
            response_format=JudgeVerdict,
        )
        message = completion.choices[0].message
        parsed = message.parsed
        if parsed is None:
            raise RuntimeError(
                f"OpenAI structured output returned no parsed result "
                f"(model={self._model}, refusal={message.refusal!r})"
            )
        return parsed
