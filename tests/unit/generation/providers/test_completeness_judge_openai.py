"""Unit tests for OpenAICompletenessJudge — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_completion(
    complete: bool = True, reasoning: str = "Addresses both parts."
) -> MagicMock:
    from src.generation.confidence_scorer import CompletenessVerdict

    completion = MagicMock()
    completion.choices[0].message.parsed = CompletenessVerdict(
        complete=complete, reasoning=reasoning
    )
    completion.choices[0].message.refusal = None
    return completion


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestOpenAICompletenessJudge:
    def test_importable(self):
        from src.generation.providers.completeness_judge_openai import (  # noqa: F401
            OpenAICompletenessJudge,
        )

    def test_judge_returns_completeness_verdict(self, settings):
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion(complete=True, reasoning="Covers both parts.")
            )
            judge = OpenAICompletenessJudge(settings)
            verdict = judge.judge(
                question="What is X and how does it compare to Y?",
                answer="X is A. Compared to Y, X is faster.",
            )

        assert verdict.complete is True
        assert verdict.reasoning == "Covers both parts."

    def test_judge_maps_incomplete_verdict(self, settings):
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion(complete=False, reasoning="Never compares to Y.")
            )
            judge = OpenAICompletenessJudge(settings)
            verdict = judge.judge(question="Question", answer="Partial answer")

        assert verdict.complete is False
        assert verdict.reasoning == "Never compares to Y."

    def test_judge_calls_sdk_with_correct_model_system_temperature(self, settings):
        from src.generation.confidence_scorer import ANSWER_COMPLETENESS_SYSTEM_PROMPT
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAICompletenessJudge(settings)
            judge.judge(question="q", answer="a")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.answer_completeness_judge_model
        assert kwargs["temperature"] == settings.answer_completeness_judge_temperature
        messages = kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == ANSWER_COMPLETENESS_SYSTEM_PROMPT

    def test_judge_builds_prompt_via_build_completeness_judge_prompt(self, settings):
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAICompletenessJudge(settings)
            judge.judge(question="What is the sky?", answer="The sky is blue.")

        kwargs = mock_parse.call_args.kwargs
        messages = kwargs["messages"]
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert "What is the sky?" in messages[1]["content"]
        assert "The sky is blue." in messages[1]["content"]
        assert "<question-" in messages[1]["content"]
        assert "<answer-" in messages[1]["content"]

    def test_judge_passes_response_format_as_completeness_verdict(self, settings):
        from src.generation.confidence_scorer import CompletenessVerdict
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAICompletenessJudge(settings)
            judge.judge(question="q", answer="a")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["response_format"] is CompletenessVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAICompletenessJudge(settings)

        assert judge.provider_id == f"openai/{settings.answer_completeness_judge_model}"

    def test_satisfies_completeness_judge_protocol(self, settings):
        from src.generation.confidence_scorer import CompletenessJudgeProtocol
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAICompletenessJudge(settings)

        assert isinstance(judge, CompletenessJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            OpenAICompletenessJudge(settings)

        MockOpenAI.assert_called_once_with(api_key=settings.openai_api_key)

    def test_openai_imported_lazily(self):
        """openai should not be imported at module top-level in the provider file."""
        import sys

        openai_mod = sys.modules.pop("openai", None)
        try:
            if "src.generation.providers.completeness_judge_openai" in sys.modules:
                del sys.modules["src.generation.providers.completeness_judge_openai"]
            import src.generation.providers.completeness_judge_openai  # noqa: F401

            assert "openai" not in sys.modules
        finally:
            if openai_mod is not None:
                sys.modules["openai"] = openai_mod

    def test_default_model_constant(self):
        from src.generation.providers.completeness_judge_openai import DEFAULT_MODEL

        assert isinstance(DEFAULT_MODEL, str)
        assert DEFAULT_MODEL.startswith("gpt-")

    def test_judge_raises_runtime_error_when_parsed_is_none(self, settings):
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            completion = MagicMock()
            completion.choices[0].message.parsed = None
            completion.choices[0].message.refusal = "I cannot assess this."
            mock_parse.return_value = completion
            judge = OpenAICompletenessJudge(settings)

            with pytest.raises(
                RuntimeError,
                match="OpenAI structured output returned no parsed result",
            ):
                judge.judge(question="q", answer="a")

    def test_judge_records_span_with_prompt_and_token_count(self, settings):
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )
        from src.tracing.context import collect_spans

        completion = _mock_completion(complete=True, reasoning="Addresses both parts.")
        completion.usage.total_tokens = 150

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = completion
            judge = OpenAICompletenessJudge(settings)
            with collect_spans() as spans:
                judge.judge(question="What is X?", answer="X is a thing.")

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "generation"
        assert recorded.token_count == 150
        assert "What is X?" in recorded.llm_prompt
        assert recorded.error is None

    def test_judge_noop_outside_collect_spans(self, settings):
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion()
            )
            judge = OpenAICompletenessJudge(settings)
            judge.judge(question="q", answer="a")
