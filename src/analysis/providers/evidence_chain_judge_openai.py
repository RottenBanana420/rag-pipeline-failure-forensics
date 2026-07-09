"""OpenAI evidence-chain narrator provider.

``openai`` is imported lazily inside ``__init__`` so this module can be
imported without the package being present. Mirrors
``failure_category_judge_openai.py``: same lazy-import convention, same
``client.chat.completions.parse(..., response_format=...)`` structured-output
call shape. The only differences are the `narrate` method name/signature and
the category-aware system prompt (`build_evidence_chain_judge_prompt`); its
own span records `step="analysis"`, matching every other judge provider in
this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis.evidence_chain import (
    EvidenceChainVerdict,
    EvidenceEntry,
    build_evidence_chain_judge_prompt,
)
from src.analysis.failure_categorizer import FailureCategory
from src.tracing.instrumentation import default_serialize, span

if TYPE_CHECKING:
    from src.config import Settings

DEFAULT_MODEL = "gpt-4o-2024-08-06"


def _extract_token_count(completion: object) -> int | None:
    usage = getattr(completion, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return total_tokens if isinstance(total_tokens, int) else None


class OpenAIEvidenceChainJudge:
    """Evidence-chain narrator backed by the OpenAI Chat Completions structured output."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # lazy import — not at module level

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.evidence_chain_judge_model
        self._temperature = settings.evidence_chain_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"openai/gpt-4o-2024-08-06"``."""
        return f"openai/{self._model}"

    def narrate(
        self,
        category: FailureCategory,
        category_rationale: str,
        chain: list[EvidenceEntry],
    ) -> EvidenceChainVerdict:
        """Synthesize a causal narrative from the ordered evidence chain."""
        prompt = build_evidence_chain_judge_prompt(category, category_rationale, chain)
        with span(
            "analysis",
            input=default_serialize(
                {
                    "category": category,
                    "category_rationale": category_rationale,
                    "chain": chain,
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
                response_format=EvidenceChainVerdict,
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
