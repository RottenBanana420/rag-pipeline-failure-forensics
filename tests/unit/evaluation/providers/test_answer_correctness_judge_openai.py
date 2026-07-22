"""Unit tests for OpenAIAnswerCorrectnessJudge — mirrors test_citation_judge_openai.py."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_completion(
    correct: bool = True, reasoning: str = "Matches expected."
) -> MagicMock:
    from src.evaluation.answer_correctness import CorrectnessVerdict

    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.parsed = CorrectnessVerdict(
        correct=correct, reasoning=reasoning
    )
    completion.choices[0].message.refusal = None
    return completion


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANSWER_CORRECTNESS_JUDGE_MODEL", "gpt-4o-2024-08-06")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestOpenAIAnswerCorrectnessJudge:
    def test_judge_returns_correctness_verdict(self, settings):
        from src.evaluation.providers.answer_correctness_judge_openai import (
            OpenAIAnswerCorrectnessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion(correct=True, reasoning="Matches.")
            )
            judge = OpenAIAnswerCorrectnessJudge(settings)
            verdict = judge.judge(
                question="Q?", expected_answer="Expected.", actual_answer="Actual."
            )

        assert verdict.correct is True
        assert verdict.reasoning == "Matches."

    def test_judge_calls_sdk_with_correct_model_system_temperature(self, settings):
        from src.evaluation.answer_correctness import (
            ANSWER_CORRECTNESS_JUDGE_SYSTEM_PROMPT,
        )
        from src.evaluation.providers.answer_correctness_judge_openai import (
            OpenAIAnswerCorrectnessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIAnswerCorrectnessJudge(settings)
            judge.judge(question="Q", expected_answer="E", actual_answer="A")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.answer_correctness_judge_model
        assert kwargs["messages"][0] == {
            "role": "system",
            "content": ANSWER_CORRECTNESS_JUDGE_SYSTEM_PROMPT,
        }
        assert kwargs["temperature"] == settings.answer_correctness_judge_temperature

    def test_judge_passes_response_format_as_correctness_verdict(self, settings):
        from src.evaluation.answer_correctness import CorrectnessVerdict
        from src.evaluation.providers.answer_correctness_judge_openai import (
            OpenAIAnswerCorrectnessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIAnswerCorrectnessJudge(settings)
            judge.judge(question="Q", expected_answer="E", actual_answer="A")

        assert mock_parse.call_args.kwargs["response_format"] is CorrectnessVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.evaluation.providers.answer_correctness_judge_openai import (
            OpenAIAnswerCorrectnessJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAIAnswerCorrectnessJudge(settings)

        assert judge.provider_id == f"openai/{settings.answer_correctness_judge_model}"

    def test_satisfies_answer_correctness_judge_protocol(self, settings):
        from src.evaluation.answer_correctness import AnswerCorrectnessJudgeProtocol
        from src.evaluation.providers.answer_correctness_judge_openai import (
            OpenAIAnswerCorrectnessJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAIAnswerCorrectnessJudge(settings)

        assert isinstance(judge, AnswerCorrectnessJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.evaluation.providers.answer_correctness_judge_openai import (
            OpenAIAnswerCorrectnessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            OpenAIAnswerCorrectnessJudge(settings)

        MockOpenAI.assert_called_once_with(api_key=settings.openai_api_key)

    def test_openai_imported_lazily(self):
        import sys

        openai_mod = sys.modules.pop("openai", None)
        try:
            sys.modules.pop(
                "src.evaluation.providers.answer_correctness_judge_openai", None
            )
            import src.evaluation.providers.answer_correctness_judge_openai  # noqa: F401

            assert "openai" not in sys.modules
        finally:
            if openai_mod is not None:
                sys.modules["openai"] = openai_mod

    def test_default_model_constant(self):
        from src.evaluation.providers.answer_correctness_judge_openai import (
            DEFAULT_MODEL,
        )

        assert DEFAULT_MODEL.startswith("gpt-")

    def test_judge_raises_runtime_error_when_parsed_is_none(self, settings):
        from src.evaluation.providers.answer_correctness_judge_openai import (
            OpenAIAnswerCorrectnessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            completion = MagicMock()
            completion.choices = [MagicMock()]
            completion.choices[0].message.parsed = None
            completion.choices[0].message.refusal = "I cannot answer that."
            MockOpenAI.return_value.chat.completions.parse.return_value = completion
            judge = OpenAIAnswerCorrectnessJudge(settings)

            with pytest.raises(
                RuntimeError, match="OpenAI structured output returned no parsed result"
            ):
                judge.judge(question="Q", expected_answer="E", actual_answer="A")

    def test_judge_records_analysis_span_with_prompt_and_token_count(self, settings):
        from src.evaluation.providers.answer_correctness_judge_openai import (
            OpenAIAnswerCorrectnessJudge,
        )
        from src.tracing.context import collect_spans

        completion = _mock_completion(correct=True, reasoning="Matches.")
        completion.usage.total_tokens = 200

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = completion
            judge = OpenAIAnswerCorrectnessJudge(settings)
            with collect_spans() as spans:
                judge.judge(
                    question="Q?", expected_answer="Expected.", actual_answer="Actual."
                )

        assert len(spans) == 1
        assert spans[0].step == "analysis"
        assert spans[0].token_count == 200
        assert spans[0].error is None
