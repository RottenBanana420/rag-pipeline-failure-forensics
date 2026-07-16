"""Anthropic answer generator provider.

``anthropic`` is imported lazily inside ``__init__`` so that this module can
be imported without the package being present. Tests should patch
``anthropic.Anthropic`` directly — Python's module cache means the patch is
visible to the inline import.

Uses the plain (non-structured-output) ``client.messages.create`` call —
confirmed via Context7 against the installed ``anthropic`` SDK (v0.116.0)
docs: a grounded answer is free-form text, not a verdict, so there is no
``output_format`` to parse against, unlike the judge providers'
``messages.parse``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.generation.prompts import GroundedPrompt
from src.tracing.instrumentation import default_serialize, span

if TYPE_CHECKING:
    from src.config import Settings

# Confirmed via Context7 against the installed anthropic SDK (v0.116.0) model
# list (anthropic.types.model.Model) as a current, non-deprecated model string —
# same default as the judge providers.
DEFAULT_MODEL = "claude-sonnet-4-5"

_MAX_TOKENS = (
    2048  # generous vs. the judges' 1024 — a grounded answer runs longer than a verdict
)


def _extract_token_count(response: object) -> int | None:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return input_tokens + output_tokens
    return None


def _extract_text(response: object) -> str:
    """Extract the text of the response's first content block.

    `response.content` is typed as a union of many block types (tool use,
    thinking, etc.), only some of which have a `.text` attribute — a plain
    `messages.create` call with no tools always returns a `TextBlock` first,
    but `getattr` (rather than a `TextBlock` isinstance check requiring an
    eager `anthropic` import) keeps this lazy-import-friendly, same rationale
    as `_extract_token_count`.
    """
    content = getattr(response, "content", [])
    first_block = content[0] if content else None
    return getattr(first_block, "text", "")


class AnthropicAnswerGenerator:
    """Grounded answer generator backed by the Anthropic Messages API."""

    def __init__(self, settings: Settings) -> None:
        from anthropic import Anthropic  # lazy import — not at module level

        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.generation_llm_model
        self._temperature = settings.generation_llm_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"anthropic/claude-sonnet-4-5"``."""
        return f"anthropic/{self._model}"

    def generate(self, prompt: GroundedPrompt) -> str:
        """Generate the grounded answer text for *prompt*."""
        with span("generation", input=default_serialize(prompt)) as s:
            s.llm_prompt = f"{prompt.system}\n\n{prompt.user}"
            response = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
                temperature=self._temperature,
            )
            s.token_count = _extract_token_count(response)
            text = _extract_text(response)
            s.output = text
            return text
