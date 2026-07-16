"""Unit tests for OpenAIAnswerGenerator — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_completion(text: str = "Paris is the capital of France [1].") -> MagicMock:
    completion = MagicMock()
    completion.choices[0].message.content = text
    return completion


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestOpenAIAnswerGenerator:
    def test_importable(self):
        from src.generation.providers.answer_generator_openai import (  # noqa: F401
            OpenAIAnswerGenerator,
        )

    def test_generate_returns_completion_text(self, settings):
        from src.generation.prompts import GroundedPrompt
        from src.generation.providers.answer_generator_openai import (
            OpenAIAnswerGenerator,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = (
                _mock_completion("The on-call rotation is weekly [1].")
            )
            generator = OpenAIAnswerGenerator(settings)
            answer = generator.generate(
                GroundedPrompt(system="system prompt", user="user prompt")
            )

        assert answer == "The on-call rotation is weekly [1]."

    def test_generate_calls_sdk_with_correct_model_system_temperature(self, settings):
        from src.generation.prompts import GroundedPrompt
        from src.generation.providers.answer_generator_openai import (
            OpenAIAnswerGenerator,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_create = MockOpenAI.return_value.chat.completions.create
            mock_create.return_value = _mock_completion()
            generator = OpenAIAnswerGenerator(settings)
            generator.generate(GroundedPrompt(system="sys text", user="user text"))

        kwargs = mock_create.call_args.kwargs
        assert kwargs["model"] == settings.generation_llm_model
        assert kwargs["temperature"] == settings.generation_llm_temperature
        assert kwargs["messages"] == [
            {"role": "system", "content": "sys text"},
            {"role": "user", "content": "user text"},
        ]

    def test_provider_id_includes_model_name(self, settings):
        from src.generation.providers.answer_generator_openai import (
            OpenAIAnswerGenerator,
        )

        with patch("openai.OpenAI"):
            generator = OpenAIAnswerGenerator(settings)

        assert generator.provider_id == f"openai/{settings.generation_llm_model}"

    def test_satisfies_answer_generator_protocol(self, settings):
        from src.generation.answer_generator import AnswerGeneratorProtocol
        from src.generation.providers.answer_generator_openai import (
            OpenAIAnswerGenerator,
        )

        with patch("openai.OpenAI"):
            generator = OpenAIAnswerGenerator(settings)

        assert isinstance(generator, AnswerGeneratorProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.generation.providers.answer_generator_openai import (
            OpenAIAnswerGenerator,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            OpenAIAnswerGenerator(settings)

        MockOpenAI.assert_called_once_with(api_key=settings.openai_api_key)

    def test_openai_imported_lazily(self):
        """openai should not be imported at module top-level in the provider file."""
        import sys

        openai_mod = sys.modules.pop("openai", None)
        try:
            if "src.generation.providers.answer_generator_openai" in sys.modules:
                del sys.modules["src.generation.providers.answer_generator_openai"]
            import src.generation.providers.answer_generator_openai  # noqa: F401

            assert "openai" not in sys.modules
        finally:
            if openai_mod is not None:
                sys.modules["openai"] = openai_mod

    def test_default_model_constant(self):
        from src.generation.providers.answer_generator_openai import DEFAULT_MODEL

        assert isinstance(DEFAULT_MODEL, str)
        assert DEFAULT_MODEL.startswith("gpt-")

    def test_generate_handles_none_content(self, settings):
        from src.generation.prompts import GroundedPrompt
        from src.generation.providers.answer_generator_openai import (
            OpenAIAnswerGenerator,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            completion = _mock_completion()
            completion.choices[0].message.content = None
            MockOpenAI.return_value.chat.completions.create.return_value = completion
            generator = OpenAIAnswerGenerator(settings)
            answer = generator.generate(GroundedPrompt(system="s", user="u"))

        assert answer == ""

    def test_generate_records_span_with_prompt_and_token_count(self, settings):
        from src.generation.prompts import GroundedPrompt
        from src.generation.providers.answer_generator_openai import (
            OpenAIAnswerGenerator,
        )
        from src.tracing.context import collect_spans

        completion = _mock_completion("The rotation is weekly [1].")
        completion.usage.total_tokens = 400

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = completion
            generator = OpenAIAnswerGenerator(settings)
            with collect_spans() as spans:
                generator.generate(GroundedPrompt(system="sys text", user="user text"))

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "generation"
        assert recorded.token_count == 400
        assert "sys text" in recorded.llm_prompt
        assert "user text" in recorded.llm_prompt
        assert recorded.output == "The rotation is weekly [1]."
        assert recorded.error is None

    def test_generate_noop_outside_collect_spans(self, settings):
        from src.generation.prompts import GroundedPrompt
        from src.generation.providers.answer_generator_openai import (
            OpenAIAnswerGenerator,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = (
                _mock_completion()
            )
            generator = OpenAIAnswerGenerator(settings)
            generator.generate(GroundedPrompt(system="s", user="u"))
