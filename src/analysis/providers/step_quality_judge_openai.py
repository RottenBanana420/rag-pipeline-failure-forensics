"""OpenAI root-cause step-quality judge provider.

``openai`` is imported lazily inside ``__init__`` so this module can be
imported without the package being present. Mirrors
``completeness_judge_openai.py``: same lazy-import convention, same
``client.chat.completions.parse(..., response_format=...)`` structured-output
call shape. The only differences are the extra `step` parameter and a
per-call, step-aware system prompt (`build_step_quality_judge_prompt`)
instead of a fixed module constant, and its own span records
`step="analysis"` rather than `"generation"`/`"verification"` — this judge
doesn't belong to any of the five original pipeline steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis.root_cause import StepQualityVerdict, build_step_quality_judge_prompt
from src.tracing.instrumentation import default_serialize, span
from src.tracing.models import PipelineStep

if TYPE_CHECKING:
    from src.config import Settings

DEFAULT_MODEL = "gpt-4o-2024-08-06"


def _extract_token_count(completion: object) -> int | None:
    usage = getattr(completion, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return total_tokens if isinstance(total_tokens, int) else None


class OpenAIStepQualityJudge:
    """Root-cause step-quality judge backed by the OpenAI Chat Completions structured output."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # lazy import — not at module level

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.root_cause_judge_model
        self._temperature = settings.root_cause_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"openai/gpt-4o-2024-08-06"``."""
        return f"openai/{self._model}"

    def judge(self, step: PipelineStep, input: str, output: str) -> StepQualityVerdict:
        """Score *step*'s input→output transformation quality, 1-5."""
        prompt = build_step_quality_judge_prompt(step, input, output)
        with span(
            "analysis",
            input=default_serialize({"step": step, "input": input, "output": output}),
        ) as s:
            s.llm_prompt = f"{prompt.system}\n\n{prompt.user}"
            completion = self._client.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                temperature=self._temperature,
                response_format=StepQualityVerdict,
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
