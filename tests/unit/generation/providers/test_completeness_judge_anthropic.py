"""Unit tests for AnthropicCompletenessJudge — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_response(
    complete: bool = True, reasoning: str = "Addresses both parts."
) -> MagicMock:
    from src.generation.confidence_scorer import CompletenessVerdict

    resp = MagicMock()
    resp.parsed_output = CompletenessVerdict(complete=complete, reasoning=reasoning)
    return resp


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestAnthropicCompletenessJudge:
    def test_importable(self):
        from src.generation.providers.completeness_judge_anthropic import (  # noqa: F401
            AnthropicCompletenessJudge,
        )

    def test_judge_returns_completeness_verdict(self, settings):
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response(
                complete=True, reasoning="Covers both parts."
            )
            judge = AnthropicCompletenessJudge(settings)
            verdict = judge.judge(
                question="What is X and how does it compare to Y?",
                answer="X is A. Compared to Y, X is faster.",
            )

        assert verdict.complete is True
        assert verdict.reasoning == "Covers both parts."

    def test_judge_maps_incomplete_verdict(self, settings):
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response(
                complete=False, reasoning="Never compares to Y."
            )
            judge = AnthropicCompletenessJudge(settings)
            verdict = judge.judge(question="Question", answer="Partial answer")

        assert verdict.complete is False
        assert verdict.reasoning == "Never compares to Y."

    def test_judge_calls_sdk_with_correct_model_system_temperature(self, settings):
        from src.generation.confidence_scorer import ANSWER_COMPLETENESS_SYSTEM_PROMPT
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicCompletenessJudge(settings)
            judge.judge(question="q", answer="a")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.answer_completeness_judge_model
        assert kwargs["system"] == ANSWER_COMPLETENESS_SYSTEM_PROMPT
        assert kwargs["temperature"] == settings.answer_completeness_judge_temperature

    def test_judge_builds_prompt_via_build_completeness_judge_prompt(self, settings):
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicCompletenessJudge(settings)
            judge.judge(question="What is the sky?", answer="The sky is blue.")

        kwargs = mock_parse.call_args.kwargs
        messages = kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "What is the sky?" in messages[0]["content"]
        assert "The sky is blue." in messages[0]["content"]
        assert "<question-" in messages[0]["content"]
        assert "<answer-" in messages[0]["content"]

    def test_judge_passes_output_format_as_completeness_verdict(self, settings):
        from src.generation.confidence_scorer import CompletenessVerdict
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicCompletenessJudge(settings)
            judge.judge(question="q", answer="a")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["output_format"] is CompletenessVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicCompletenessJudge(settings)

        assert (
            judge.provider_id == f"anthropic/{settings.answer_completeness_judge_model}"
        )

    def test_satisfies_completeness_judge_protocol(self, settings):
        from src.generation.confidence_scorer import CompletenessJudgeProtocol
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicCompletenessJudge(settings)

        assert isinstance(judge, CompletenessJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            AnthropicCompletenessJudge(settings)

        MockAnthropic.assert_called_once_with(api_key=settings.anthropic_api_key)

    def test_anthropic_imported_lazily(self):
        """anthropic should not be imported at module top-level in the provider file."""
        import sys

        anthropic_mod = sys.modules.pop("anthropic", None)
        try:
            if "src.generation.providers.completeness_judge_anthropic" in sys.modules:
                del sys.modules["src.generation.providers.completeness_judge_anthropic"]
            import src.generation.providers.completeness_judge_anthropic  # noqa: F401

            assert "anthropic" not in sys.modules
        finally:
            if anthropic_mod is not None:
                sys.modules["anthropic"] = anthropic_mod

    def test_default_model_constant(self):
        from src.generation.providers.completeness_judge_anthropic import DEFAULT_MODEL

        assert isinstance(DEFAULT_MODEL, str)
        assert DEFAULT_MODEL.startswith("claude-")

    def test_judge_raises_runtime_error_when_parsed_output_is_none(self, settings):
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            resp = MagicMock()
            resp.parsed_output = None
            mock_parse.return_value = resp
            judge = AnthropicCompletenessJudge(settings)

            with pytest.raises(
                RuntimeError,
                match="Anthropic structured output returned no parsed_output",
            ):
                judge.judge(question="q", answer="a")

    def test_judge_records_span_with_prompt_and_token_count(self, settings):
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )
        from src.tracing.context import collect_spans

        resp = _mock_response(complete=True, reasoning="Addresses both parts.")
        resp.usage.input_tokens = 80
        resp.usage.output_tokens = 20

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = resp
            judge = AnthropicCompletenessJudge(settings)
            with collect_spans() as spans:
                judge.judge(question="What is X?", answer="X is a thing.")

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "generation"
        assert recorded.token_count == 100
        assert "What is X?" in recorded.llm_prompt
        assert recorded.error is None

    def test_judge_noop_outside_collect_spans(self, settings):
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response()
            judge = AnthropicCompletenessJudge(settings)
            judge.judge(question="q", answer="a")
