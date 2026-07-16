"""Unit tests for AnthropicAnswerGenerator — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_response(text: str = "Paris is the capital of France [1].") -> MagicMock:
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestAnthropicAnswerGenerator:
    def test_importable(self):
        from src.generation.providers.answer_generator_anthropic import (  # noqa: F401
            AnthropicAnswerGenerator,
        )

    def test_generate_returns_response_text(self, settings):
        from src.generation.prompts import GroundedPrompt
        from src.generation.providers.answer_generator_anthropic import (
            AnthropicAnswerGenerator,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = _mock_response(
                "The on-call rotation is weekly [1]."
            )
            generator = AnthropicAnswerGenerator(settings)
            answer = generator.generate(
                GroundedPrompt(system="system prompt", user="user prompt")
            )

        assert answer == "The on-call rotation is weekly [1]."

    def test_generate_calls_sdk_with_correct_model_system_temperature(self, settings):
        from src.generation.prompts import GroundedPrompt
        from src.generation.providers.answer_generator_anthropic import (
            AnthropicAnswerGenerator,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_create = MockAnthropic.return_value.messages.create
            mock_create.return_value = _mock_response()
            generator = AnthropicAnswerGenerator(settings)
            generator.generate(GroundedPrompt(system="sys text", user="user text"))

        kwargs = mock_create.call_args.kwargs
        assert kwargs["model"] == settings.generation_llm_model
        assert kwargs["system"] == "sys text"
        assert kwargs["temperature"] == settings.generation_llm_temperature
        assert kwargs["messages"] == [{"role": "user", "content": "user text"}]

    def test_provider_id_includes_model_name(self, settings):
        from src.generation.providers.answer_generator_anthropic import (
            AnthropicAnswerGenerator,
        )

        with patch("anthropic.Anthropic"):
            generator = AnthropicAnswerGenerator(settings)

        assert generator.provider_id == f"anthropic/{settings.generation_llm_model}"

    def test_satisfies_answer_generator_protocol(self, settings):
        from src.generation.answer_generator import AnswerGeneratorProtocol
        from src.generation.providers.answer_generator_anthropic import (
            AnthropicAnswerGenerator,
        )

        with patch("anthropic.Anthropic"):
            generator = AnthropicAnswerGenerator(settings)

        assert isinstance(generator, AnswerGeneratorProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.generation.providers.answer_generator_anthropic import (
            AnthropicAnswerGenerator,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            AnthropicAnswerGenerator(settings)

        MockAnthropic.assert_called_once_with(api_key=settings.anthropic_api_key)

    def test_anthropic_imported_lazily(self):
        """anthropic should not be imported at module top-level in the provider file."""
        import sys

        anthropic_mod = sys.modules.pop("anthropic", None)
        try:
            if "src.generation.providers.answer_generator_anthropic" in sys.modules:
                del sys.modules["src.generation.providers.answer_generator_anthropic"]
            import src.generation.providers.answer_generator_anthropic  # noqa: F401

            assert "anthropic" not in sys.modules
        finally:
            if anthropic_mod is not None:
                sys.modules["anthropic"] = anthropic_mod

    def test_default_model_constant(self):
        from src.generation.providers.answer_generator_anthropic import DEFAULT_MODEL

        assert isinstance(DEFAULT_MODEL, str)
        assert DEFAULT_MODEL.startswith("claude-")

    def test_generate_records_span_with_prompt_and_token_count(self, settings):
        from src.generation.prompts import GroundedPrompt
        from src.generation.providers.answer_generator_anthropic import (
            AnthropicAnswerGenerator,
        )
        from src.tracing.context import collect_spans

        resp = _mock_response("The rotation is weekly [1].")
        resp.usage.input_tokens = 300
        resp.usage.output_tokens = 40

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = resp
            generator = AnthropicAnswerGenerator(settings)
            with collect_spans() as spans:
                generator.generate(GroundedPrompt(system="sys text", user="user text"))

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "generation"
        assert recorded.token_count == 340
        assert "sys text" in recorded.llm_prompt
        assert "user text" in recorded.llm_prompt
        assert recorded.output == "The rotation is weekly [1]."
        assert recorded.error is None

    def test_generate_noop_outside_collect_spans(self, settings):
        from src.generation.prompts import GroundedPrompt
        from src.generation.providers.answer_generator_anthropic import (
            AnthropicAnswerGenerator,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = _mock_response()
            generator = AnthropicAnswerGenerator(settings)
            generator.generate(GroundedPrompt(system="s", user="u"))

    def test_generate_returns_empty_string_for_non_text_first_block(self, settings):
        """Regression test for _extract_text's getattr fallback: a plain
        object() (not MagicMock, which auto-vivifies .text and would hide
        this bug) simulates a real anthropic ToolUseBlock/ThinkingBlock with
        no `.text` attribute as content[0] — should degrade to "", not raise."""
        from src.generation.prompts import GroundedPrompt
        from src.generation.providers.answer_generator_anthropic import (
            AnthropicAnswerGenerator,
        )

        non_text_block = object()
        resp = MagicMock()
        resp.content = [non_text_block]

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = resp
            generator = AnthropicAnswerGenerator(settings)
            answer = generator.generate(GroundedPrompt(system="s", user="u"))

        assert answer == ""

    def test_generate_returns_empty_string_for_empty_content(self, settings):
        """Same fallback, for the (unlikely but possible) empty-content-list case."""
        from src.generation.prompts import GroundedPrompt
        from src.generation.providers.answer_generator_anthropic import (
            AnthropicAnswerGenerator,
        )

        resp = MagicMock()
        resp.content = []

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = resp
            generator = AnthropicAnswerGenerator(settings)
            answer = generator.generate(GroundedPrompt(system="s", user="u"))

        assert answer == ""
