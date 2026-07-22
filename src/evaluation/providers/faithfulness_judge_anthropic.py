"""Anthropic faithfulness judge provider.

``anthropic`` is imported lazily inside ``__init__`` so that this module can
be imported without the package being present. Tests should patch
``anthropic.Anthropic`` directly — Python's module cache means the patch is
visible to the inline import.

Uses the stable (non-beta) ``client.messages.parse(..., output_format=...)``
structured-output API — same call shape as
``src.generation.providers.citation_judge_anthropic``, re-confirmed via
Context7 against the installed ``anthropic`` SDK (v0.116.0): the
``output_format`` -> ``output_config.format`` deprecation warning only exists
on ``anthropic.resources.beta.messages.messages``, not the top-level
(non-beta) ``client.messages.parse`` this module calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.evaluation.faithfulness import (
    FAITHFULNESS_JUDGE_SYSTEM_PROMPT,
    FaithfulnessVerdict,
    build_faithfulness_judge_prompt,
)
from src.tracing.instrumentation import default_serialize, span

if TYPE_CHECKING:
    from src.config import Settings

# Confirmed via Context7 against the installed anthropic SDK (v0.116.0)
# model list (anthropic.types.model.Model) as a current, non-deprecated
# model string.
DEFAULT_MODEL = "claude-sonnet-4-5"


def _extract_token_count(response: object) -> int | None:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return input_tokens + output_tokens
    return None


class AnthropicFaithfulnessJudge:
    """Faithfulness judge backed by the Anthropic Messages API structured output."""

    def __init__(self, settings: Settings) -> None:
        from anthropic import Anthropic  # lazy import — not at module level

        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.faithfulness_judge_model
        self._temperature = settings.faithfulness_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"anthropic/claude-sonnet-4-5"``."""
        return f"anthropic/{self._model}"

    def judge(self, claim: str, context: str) -> FaithfulnessVerdict:
        """Decide whether *context* grounds *claim* and return a verdict."""
        prompt = build_faithfulness_judge_prompt(claim, context)
        with span(
            "verification",
            input=default_serialize({"claim": claim, "context": context}),
        ) as s:
            s.llm_prompt = f"{FAITHFULNESS_JUDGE_SYSTEM_PROMPT}\n\n{prompt.user}"
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=1024,
                system=FAITHFULNESS_JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt.user}],
                temperature=self._temperature,
                output_format=FaithfulnessVerdict,
            )
            s.token_count = _extract_token_count(response)
            parsed = response.parsed_output
            if parsed is None:
                raise RuntimeError(
                    f"Anthropic structured output returned no parsed_output "
                    f"(model={self._model})"
                )
            s.output = default_serialize(parsed)
            return parsed
