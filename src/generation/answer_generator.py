"""Grounded answer generator — protocol, verdict-free LLM call, and provider factory.

No other module calls an LLM to produce the actual `[N]`-cited answer text —
`prompts.py` only builds the prompt (`build_grounded_prompt`). This module fills
that gap with `make_answer_generator`, the same lazy-import factory pattern as
`make_citation_judge`/`make_completeness_judge`/`make_reranker`/`make_embedder`.

Unlike the judge protocols (`CitationJudgeProtocol`, `CompletenessJudgeProtocol`),
`generate` returns plain text, not a structured verdict — an answer is free-form
prose, not a boolean decision — so providers use their SDK's plain
`messages.create`/`chat.completions.create` call, not `.parse`/structured output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.generation.prompts import GroundedPrompt

if TYPE_CHECKING:
    from src.config import Settings


@runtime_checkable
class AnswerGeneratorProtocol(Protocol):
    """Structural interface every answer-generation provider must satisfy."""

    def generate(self, prompt: GroundedPrompt) -> str:
        """Generate the grounded answer text for *prompt*."""
        ...

    @property
    def provider_id(self) -> str:
        """Short identifier for the provider, e.g. ``"anthropic/claude-sonnet-4-5"``."""
        ...


def make_answer_generator(settings: Settings) -> AnswerGeneratorProtocol:
    """Return an answer generator instance for the provider in *settings*.

    Provider modules are imported lazily inside this function so that importing
    ``src.generation.answer_generator`` does not pull in optional heavy
    dependencies (e.g. the ``anthropic`` or ``openai`` SDKs) unless they are
    actually needed. Mirrors ``make_citation_judge``/``make_completeness_judge``.

    Raises:
        ValueError: If ``settings.generation_llm_provider`` is not a recognised value.
    """
    provider = settings.generation_llm_provider

    if provider == "anthropic":
        from src.generation.providers.answer_generator_anthropic import (
            DEFAULT_MODEL as _ANTHROPIC_DEFAULT_MODEL,
        )
        from src.generation.providers.answer_generator_anthropic import (
            AnthropicAnswerGenerator as _AnthropicAnswerGenerator,
        )

        model_name = (
            settings.generation_llm_model
            if settings.generation_llm_model.startswith("claude")
            else _ANTHROPIC_DEFAULT_MODEL
        )
        return _AnthropicAnswerGenerator(
            settings.model_copy(update={"generation_llm_model": model_name})
        )

    if provider == "openai":
        from src.generation.providers.answer_generator_openai import (
            DEFAULT_MODEL as _OPENAI_DEFAULT_MODEL,
        )
        from src.generation.providers.answer_generator_openai import (
            OpenAIAnswerGenerator as _OpenAIAnswerGenerator,
        )

        model_name = (
            settings.generation_llm_model
            if settings.generation_llm_model.startswith("gpt")
            else _OPENAI_DEFAULT_MODEL
        )
        return _OpenAIAnswerGenerator(
            settings.model_copy(update={"generation_llm_model": model_name})
        )

    valid = "anthropic, openai"
    raise ValueError(
        f"Unknown generation LLM provider: {provider!r}. Valid providers are: {valid}"
    )
