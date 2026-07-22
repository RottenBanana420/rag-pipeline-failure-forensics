"""Unit tests for AnthropicAnswerCorrectnessJudge — mirrors test_citation_judge_anthropic.py."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_response(
    correct: bool = True, reasoning: str = "Matches expected."
) -> MagicMock:
    from src.evaluation.answer_correctness import CorrectnessVerdict

    resp = MagicMock()
    resp.parsed_output = CorrectnessVerdict(correct=correct, reasoning=reasoning)
    return resp


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestAnthropicAnswerCorrectnessJudge:
    def test_judge_returns_correctness_verdict(self, settings):
        from src.evaluation.providers.answer_correctness_judge_anthropic import (
            AnthropicAnswerCorrectnessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response(
                correct=True, reasoning="Matches."
            )
            judge = AnthropicAnswerCorrectnessJudge(settings)
            verdict = judge.judge(
                question="Q?", expected_answer="Expected.", actual_answer="Actual."
            )

        assert verdict.correct is True
        assert verdict.reasoning == "Matches."

    def test_judge_calls_sdk_with_correct_model_system_temperature(self, settings):
        from src.evaluation.answer_correctness import (
            ANSWER_CORRECTNESS_JUDGE_SYSTEM_PROMPT,
        )
        from src.evaluation.providers.answer_correctness_judge_anthropic import (
            AnthropicAnswerCorrectnessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicAnswerCorrectnessJudge(settings)
            judge.judge(question="Q", expected_answer="E", actual_answer="A")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.answer_correctness_judge_model
        assert kwargs["system"] == ANSWER_CORRECTNESS_JUDGE_SYSTEM_PROMPT
        assert kwargs["temperature"] == settings.answer_correctness_judge_temperature

    def test_judge_passes_output_format_as_correctness_verdict(self, settings):
        from src.evaluation.answer_correctness import CorrectnessVerdict
        from src.evaluation.providers.answer_correctness_judge_anthropic import (
            AnthropicAnswerCorrectnessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicAnswerCorrectnessJudge(settings)
            judge.judge(question="Q", expected_answer="E", actual_answer="A")

        assert mock_parse.call_args.kwargs["output_format"] is CorrectnessVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.evaluation.providers.answer_correctness_judge_anthropic import (
            AnthropicAnswerCorrectnessJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicAnswerCorrectnessJudge(settings)

        assert (
            judge.provider_id == f"anthropic/{settings.answer_correctness_judge_model}"
        )

    def test_satisfies_answer_correctness_judge_protocol(self, settings):
        from src.evaluation.answer_correctness import AnswerCorrectnessJudgeProtocol
        from src.evaluation.providers.answer_correctness_judge_anthropic import (
            AnthropicAnswerCorrectnessJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicAnswerCorrectnessJudge(settings)

        assert isinstance(judge, AnswerCorrectnessJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.evaluation.providers.answer_correctness_judge_anthropic import (
            AnthropicAnswerCorrectnessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            AnthropicAnswerCorrectnessJudge(settings)

        MockAnthropic.assert_called_once_with(api_key=settings.anthropic_api_key)

    def test_anthropic_imported_lazily(self):
        import sys

        anthropic_mod = sys.modules.pop("anthropic", None)
        try:
            sys.modules.pop(
                "src.evaluation.providers.answer_correctness_judge_anthropic", None
            )
            import src.evaluation.providers.answer_correctness_judge_anthropic  # noqa: F401

            assert "anthropic" not in sys.modules
        finally:
            if anthropic_mod is not None:
                sys.modules["anthropic"] = anthropic_mod

    def test_default_model_constant(self):
        from src.evaluation.providers.answer_correctness_judge_anthropic import (
            DEFAULT_MODEL,
        )

        assert DEFAULT_MODEL.startswith("claude-")

    def test_judge_raises_runtime_error_when_parsed_output_is_none(self, settings):
        from src.evaluation.providers.answer_correctness_judge_anthropic import (
            AnthropicAnswerCorrectnessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            resp = MagicMock()
            resp.parsed_output = None
            MockAnthropic.return_value.messages.parse.return_value = resp
            judge = AnthropicAnswerCorrectnessJudge(settings)

            with pytest.raises(
                RuntimeError,
                match="Anthropic structured output returned no parsed_output",
            ):
                judge.judge(question="Q", expected_answer="E", actual_answer="A")

    def test_judge_records_analysis_span_with_prompt_and_token_count(self, settings):
        from src.evaluation.providers.answer_correctness_judge_anthropic import (
            AnthropicAnswerCorrectnessJudge,
        )
        from src.tracing.context import collect_spans

        resp = _mock_response(correct=True, reasoning="Matches.")
        resp.usage.input_tokens = 80
        resp.usage.output_tokens = 15

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = resp
            judge = AnthropicAnswerCorrectnessJudge(settings)
            with collect_spans() as spans:
                judge.judge(
                    question="Q?", expected_answer="Expected.", actual_answer="Actual."
                )

        assert len(spans) == 1
        assert spans[0].step == "analysis"
        assert spans[0].token_count == 95
        assert spans[0].error is None
