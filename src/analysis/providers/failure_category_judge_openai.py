"""OpenAI failure-category judge provider.

``openai`` is imported lazily inside ``__init__`` so this module can be
imported without the package being present. Mirrors
``step_quality_judge_openai.py``: same lazy-import convention, same
``client.chat.completions.parse(..., response_format=...)`` structured-output
call shape. The only differences are the extra `quality_rationale` parameter
and the taxonomy-aware system prompt (`build_failure_category_judge_prompt`)
instead of a fixed module constant; its own span records `step="analysis"`,
matching the step-quality judge providers.
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

DEFAULT_MODEL = "gpt-4o-2024-08-06"


def _extract_token_count(completion: object) -> int | None:
    usage = getattr(completion, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return total_tokens if isinstance(total_tokens, int) else None


class OpenAIFailureCategoryJudge:
    """Failure-category judge backed by the OpenAI Chat Completions structured output."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # lazy import — not at module level

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.failure_category_judge_model
        self._temperature = settings.failure_category_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"openai/gpt-4o-2024-08-06"``."""
        return f"openai/{self._model}"

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
            completion = self._client.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                temperature=self._temperature,
                response_format=FailureCategoryVerdict,
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
