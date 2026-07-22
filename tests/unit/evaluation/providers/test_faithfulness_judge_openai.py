"""Unit tests for OpenAIFaithfulnessJudge — mirrors test_citation_judge_openai.py."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_completion(
    grounded: bool = True, reasoning: str = "Grounded in context."
) -> MagicMock:
    from src.evaluation.faithfulness import FaithfulnessVerdict

    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.parsed = FaithfulnessVerdict(
        grounded=grounded, reasoning=reasoning
    )
    completion.choices[0].message.refusal = None
    return completion


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("FAITHFULNESS_JUDGE_MODEL", "gpt-4o-2024-08-06")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestOpenAIFaithfulnessJudge:
    def test_judge_returns_faithfulness_verdict(self, settings):
        from src.evaluation.providers.faithfulness_judge_openai import (
            OpenAIFaithfulnessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion(grounded=True, reasoning="Matches context.")
            )
            judge = OpenAIFaithfulnessJudge(settings)
            verdict = judge.judge(
                claim="The sky is blue.", context="The sky appears blue."
            )

        assert verdict.grounded is True
        assert verdict.reasoning == "Matches context."

    def test_judge_calls_sdk_with_correct_model_system_temperature(self, settings):
        from src.evaluation.faithfulness import FAITHFULNESS_JUDGE_SYSTEM_PROMPT
        from src.evaluation.providers.faithfulness_judge_openai import (
            OpenAIFaithfulnessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIFaithfulnessJudge(settings)
            judge.judge(claim="c", context="e")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.faithfulness_judge_model
        assert kwargs["messages"][0] == {
            "role": "system",
            "content": FAITHFULNESS_JUDGE_SYSTEM_PROMPT,
        }
        assert kwargs["temperature"] == settings.faithfulness_judge_temperature

    def test_judge_passes_response_format_as_faithfulness_verdict(self, settings):
        from src.evaluation.faithfulness import FaithfulnessVerdict
        from src.evaluation.providers.faithfulness_judge_openai import (
            OpenAIFaithfulnessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIFaithfulnessJudge(settings)
            judge.judge(claim="c", context="e")

        assert mock_parse.call_args.kwargs["response_format"] is FaithfulnessVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.evaluation.providers.faithfulness_judge_openai import (
            OpenAIFaithfulnessJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAIFaithfulnessJudge(settings)

        assert judge.provider_id == f"openai/{settings.faithfulness_judge_model}"

    def test_satisfies_faithfulness_judge_protocol(self, settings):
        from src.evaluation.faithfulness import FaithfulnessJudgeProtocol
        from src.evaluation.providers.faithfulness_judge_openai import (
            OpenAIFaithfulnessJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAIFaithfulnessJudge(settings)

        assert isinstance(judge, FaithfulnessJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.evaluation.providers.faithfulness_judge_openai import (
            OpenAIFaithfulnessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            OpenAIFaithfulnessJudge(settings)

        MockOpenAI.assert_called_once_with(api_key=settings.openai_api_key)

    def test_openai_imported_lazily(self):
        import sys

        openai_mod = sys.modules.pop("openai", None)
        try:
            sys.modules.pop("src.evaluation.providers.faithfulness_judge_openai", None)
            import src.evaluation.providers.faithfulness_judge_openai  # noqa: F401

            assert "openai" not in sys.modules
        finally:
            if openai_mod is not None:
                sys.modules["openai"] = openai_mod

    def test_default_model_constant(self):
        from src.evaluation.providers.faithfulness_judge_openai import DEFAULT_MODEL

        assert DEFAULT_MODEL.startswith("gpt-")

    def test_judge_raises_runtime_error_when_parsed_is_none(self, settings):
        from src.evaluation.providers.faithfulness_judge_openai import (
            OpenAIFaithfulnessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            completion = MagicMock()
            completion.choices = [MagicMock()]
            completion.choices[0].message.parsed = None
            completion.choices[0].message.refusal = "I cannot answer that."
            MockOpenAI.return_value.chat.completions.parse.return_value = completion
            judge = OpenAIFaithfulnessJudge(settings)

            with pytest.raises(
                RuntimeError, match="OpenAI structured output returned no parsed result"
            ):
                judge.judge(claim="c", context="e")

    def test_judge_records_span_with_prompt_and_token_count(self, settings):
        from src.evaluation.providers.faithfulness_judge_openai import (
            OpenAIFaithfulnessJudge,
        )
        from src.tracing.context import collect_spans

        completion = _mock_completion(grounded=True, reasoning="Matches.")
        completion.usage.total_tokens = 150

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = completion
            judge = OpenAIFaithfulnessJudge(settings)
            with collect_spans() as spans:
                judge.judge(claim="The sky is blue.", context="The sky appears blue.")

        assert len(spans) == 1
        assert spans[0].step == "verification"
        assert spans[0].token_count == 150
        assert spans[0].error is None
