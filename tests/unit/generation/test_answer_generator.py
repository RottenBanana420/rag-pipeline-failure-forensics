"""Unit tests for the answer generator core (protocol + make_answer_generator factory)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def anthropic_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GENERATION_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


@pytest.fixture
def openai_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GENERATION_LLM_PROVIDER", "openai")
    monkeypatch.setenv("GENERATION_LLM_MODEL", "gpt-4o-2024-08-06")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestMakeAnswerGenerator:
    def test_importable(self):
        from src.generation.answer_generator import make_answer_generator  # noqa: F401

    def test_anthropic_provider_returns_anthropic_generator(self, anthropic_settings):
        from src.generation.answer_generator import make_answer_generator
        from src.generation.providers.answer_generator_anthropic import (
            AnthropicAnswerGenerator,
        )

        with patch("anthropic.Anthropic", return_value=MagicMock()):
            result = make_answer_generator(anthropic_settings)

        assert isinstance(result, AnthropicAnswerGenerator)

    def test_anthropic_provider_id_reflects_resolved_model(self, anthropic_settings):
        from src.generation.answer_generator import make_answer_generator

        assert anthropic_settings.generation_llm_model == "claude-sonnet-4-5"

        with patch("anthropic.Anthropic", return_value=MagicMock()):
            result = make_answer_generator(anthropic_settings)

        assert result.provider_id == "anthropic/claude-sonnet-4-5"

    def test_anthropic_provider_substitutes_default_when_model_not_claude(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GENERATION_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("GENERATION_LLM_MODEL", "gpt-4o-2024-08-06")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.generation.answer_generator import make_answer_generator
        from src.generation.providers.answer_generator_anthropic import DEFAULT_MODEL

        settings = Settings()
        assert not settings.generation_llm_model.startswith("claude")

        with patch("anthropic.Anthropic", return_value=MagicMock()):
            result = make_answer_generator(settings)

        assert result.provider_id == f"anthropic/{DEFAULT_MODEL}"

    def test_openai_provider_returns_openai_generator(self, openai_settings):
        from src.generation.answer_generator import make_answer_generator
        from src.generation.providers.answer_generator_openai import (
            OpenAIAnswerGenerator,
        )

        with patch("openai.OpenAI", return_value=MagicMock()):
            result = make_answer_generator(openai_settings)

        assert isinstance(result, OpenAIAnswerGenerator)

    def test_openai_provider_id_reflects_resolved_model(self, openai_settings):
        from src.generation.answer_generator import make_answer_generator

        with patch("openai.OpenAI", return_value=MagicMock()):
            result = make_answer_generator(openai_settings)

        assert result.provider_id == "openai/gpt-4o-2024-08-06"

    def test_openai_provider_substitutes_default_when_model_not_gpt(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("GENERATION_LLM_PROVIDER", "openai")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.generation.answer_generator import make_answer_generator
        from src.generation.providers.answer_generator_openai import DEFAULT_MODEL

        settings = Settings()
        assert not settings.generation_llm_model.startswith("gpt")

        with patch("openai.OpenAI", return_value=MagicMock()):
            result = make_answer_generator(settings)

        assert result.provider_id == f"openai/{DEFAULT_MODEL}"

    def test_unknown_provider_raises_value_error(self, anthropic_settings):
        from src.generation.answer_generator import make_answer_generator

        object.__setattr__(
            anthropic_settings, "generation_llm_provider", "unsupported_provider"
        )

        with pytest.raises(ValueError, match="unsupported_provider"):
            make_answer_generator(anthropic_settings)

    def test_unknown_provider_error_lists_valid_providers(self, anthropic_settings):
        from src.generation.answer_generator import make_answer_generator

        object.__setattr__(anthropic_settings, "generation_llm_provider", "bogus")

        with pytest.raises(ValueError) as exc_info:
            make_answer_generator(anthropic_settings)

        assert "anthropic" in str(exc_info.value)
        assert "openai" in str(exc_info.value)

    def test_anthropic_result_satisfies_answer_generator_protocol(
        self, anthropic_settings
    ):
        from src.generation.answer_generator import (
            AnswerGeneratorProtocol,
            make_answer_generator,
        )

        with patch("anthropic.Anthropic", return_value=MagicMock()):
            result = make_answer_generator(anthropic_settings)

        assert isinstance(result, AnswerGeneratorProtocol)

    def test_openai_result_satisfies_answer_generator_protocol(self, openai_settings):
        from src.generation.answer_generator import (
            AnswerGeneratorProtocol,
            make_answer_generator,
        )

        with patch("openai.OpenAI", return_value=MagicMock()):
            result = make_answer_generator(openai_settings)

        assert isinstance(result, AnswerGeneratorProtocol)
