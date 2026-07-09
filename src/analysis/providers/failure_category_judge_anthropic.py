"""Anthropic failure-category judge provider.

``anthropic`` is imported lazily inside ``__init__`` so this module can be
imported without the package being present. Mirrors
``step_quality_judge_anthropic.py``: same lazy-import convention, same
``messages.parse(..., output_format=...)`` structured-output call shape. The
only differences are the extra `quality_rationale` parameter and the
taxonomy-aware system prompt (`build_failure_category_judge_prompt`) instead
of a fixed module constant; its own span records `step="analysis"`, matching
the step-quality judge providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis.failure_categorizer import (
    FailureCategoryVerdict,
    build_failure_category_judge_prompt,
)
from src.tracing.instrumentation import default_serialize, span
from src.tracing.models import PipelineStep

if TYPE_CHECKING:
    from src.config import Settings

DEFAULT_MODEL = "claude-sonnet-4-5"


def _extract_token_count(response: object) -> int | None:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return input_tokens + output_tokens
    return None


class AnthropicFailureCategoryJudge:
    """Failure-category judge backed by the Anthropic Messages API structured output."""

    def __init__(self, settings: Settings) -> None:
        from anthropic import Anthropic  # lazy import — not at module level

        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.failure_category_judge_model
        self._temperature = settings.failure_category_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"anthropic/claude-sonnet-4-5"``."""
        return f"anthropic/{self._model}"

    def classify(
        self, step: PipelineStep, input: str, output: str, quality_rationale: str
    ) -> FailureCategoryVerdict:
        """Classify the failure represented by *step*'s input→output transformation."""
        prompt = build_failure_category_judge_prompt(
            step, input, output, quality_rationale
        )
        with span(
            "analysis",
            input=default_serialize(
                {
                    "step": step,
                    "input": input,
                    "output": output,
                    "quality_rationale": quality_rationale,
                }
            ),
        ) as s:
            s.llm_prompt = f"{prompt.system}\n\n{prompt.user}"
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=1024,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
                temperature=self._temperature,
                output_format=FailureCategoryVerdict,
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
