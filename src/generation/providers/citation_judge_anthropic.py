"""Anthropic citation judge provider.

``anthropic`` is imported lazily inside ``__init__`` so that this module can
be imported without the package being present. Tests should patch
``anthropic.Anthropic`` directly — Python's module cache means the patch is
visible to the inline import.

Uses the stable (non-beta) ``client.messages.parse(..., output_format=...)``
structured-output API. Confirmed via Context7 against the installed
``anthropic`` SDK source (v0.116.0): the ``output_format`` -> ``output_config
.format`` deprecation warning (``_warn_output_format_deprecated``) only
exists in ``anthropic.resources.beta.messages.messages`` — the top-level
``client.messages.parse`` merges ``output_format`` into ``output_config``
internally without any deprecation warning, so ``output_format`` is the
correct, current parameter for this non-beta call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.generation.citation_verifier import (
    CITATION_JUDGE_SYSTEM_PROMPT,
    JudgeVerdict,
    build_judge_prompt,
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


class AnthropicCitationJudge:
    """Citation judge backed by the Anthropic Messages API structured output."""

    def __init__(self, settings: Settings) -> None:
        from anthropic import Anthropic  # lazy import — not at module level

        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.citation_judge_model
        self._temperature = settings.citation_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"anthropic/claude-sonnet-4-5"``."""
        return f"anthropic/{self._model}"

    def judge(self, claim: str, evidence: str) -> JudgeVerdict:
        """Decide whether *evidence* supports *claim* and return a verdict."""
        prompt = build_judge_prompt(claim, evidence)
        with span(
            "verification",
            input=default_serialize({"claim": claim, "evidence": evidence}),
        ) as s:
            s.llm_prompt = f"{CITATION_JUDGE_SYSTEM_PROMPT}\n\n{prompt.user}"
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=1024,
                system=CITATION_JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt.user}],
                temperature=self._temperature,
                output_format=JudgeVerdict,
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
