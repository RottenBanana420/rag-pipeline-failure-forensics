"""OpenAI answer generator provider.

``openai`` is imported lazily inside ``__init__`` so that this module can be
imported without the package being present. Tests should patch
``openai.OpenAI`` directly — Python's module cache means the patch is
visible to the inline import.

Uses the plain (non-structured-output) ``client.chat.completions.create``
call — confirmed via Context7 against the installed ``openai`` SDK (v2.44.0)
docs: a grounded answer is free-form text, not a verdict, so there is no
``response_format`` to parse against, unlike the judge providers'
``chat.completions.parse``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.generation.prompts import GroundedPrompt
from src.tracing.instrumentation import default_serialize, span

if TYPE_CHECKING:
    from src.config import Settings

# Confirmed via Context7 (openai-python docs) and cross-checked against the
# installed openai SDK's ChatModel literal (v2.44.0) — same default as the
# judge providers.
DEFAULT_MODEL = "gpt-4o-2024-08-06"


def _extract_token_count(completion: object) -> int | None:
    usage = getattr(completion, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return total_tokens if isinstance(total_tokens, int) else None


class OpenAIAnswerGenerator:
    """Grounded answer generator backed by the OpenAI Chat Completions API."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # lazy import — not at module level

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.generation_llm_model
        self._temperature = settings.generation_llm_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"openai/gpt-4o-2024-08-06"``."""
        return f"openai/{self._model}"

    def generate(self, prompt: GroundedPrompt) -> str:
        """Generate the grounded answer text for *prompt*."""
        with span("generation", input=default_serialize(prompt)) as s:
            s.llm_prompt = f"{prompt.system}\n\n{prompt.user}"
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                temperature=self._temperature,
            )
            s.token_count = _extract_token_count(completion)
            text = completion.choices[0].message.content or ""
            s.output = text
            return text
