"""Unit tests for AnthropicFaithfulnessJudge — mirrors test_citation_judge_anthropic.py."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_response(
    grounded: bool = True, reasoning: str = "Grounded in context."
) -> MagicMock:
    from src.evaluation.faithfulness import FaithfulnessVerdict

    resp = MagicMock()
    resp.parsed_output = FaithfulnessVerdict(grounded=grounded, reasoning=reasoning)
    return resp


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestAnthropicFaithfulnessJudge:
    def test_judge_returns_faithfulness_verdict(self, settings):
        from src.evaluation.providers.faithfulness_judge_anthropic import (
            AnthropicFaithfulnessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response(
                grounded=True, reasoning="Matches context."
            )
            judge = AnthropicFaithfulnessJudge(settings)
            verdict = judge.judge(
                claim="The sky is blue.", context="The sky appears blue."
            )

        assert verdict.grounded is True
        assert verdict.reasoning == "Matches context."

    def test_judge_calls_sdk_with_correct_model_system_temperature(self, settings):
        from src.evaluation.faithfulness import FAITHFULNESS_JUDGE_SYSTEM_PROMPT
        from src.evaluation.providers.faithfulness_judge_anthropic import (
            AnthropicFaithfulnessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicFaithfulnessJudge(settings)
            judge.judge(claim="c", context="e")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.faithfulness_judge_model
        assert kwargs["system"] == FAITHFULNESS_JUDGE_SYSTEM_PROMPT
        assert kwargs["temperature"] == settings.faithfulness_judge_temperature

    def test_judge_passes_output_format_as_faithfulness_verdict(self, settings):
        from src.evaluation.faithfulness import FaithfulnessVerdict
        from src.evaluation.providers.faithfulness_judge_anthropic import (
            AnthropicFaithfulnessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicFaithfulnessJudge(settings)
            judge.judge(claim="c", context="e")

        assert mock_parse.call_args.kwargs["output_format"] is FaithfulnessVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.evaluation.providers.faithfulness_judge_anthropic import (
            AnthropicFaithfulnessJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicFaithfulnessJudge(settings)

        assert judge.provider_id == f"anthropic/{settings.faithfulness_judge_model}"

    def test_satisfies_faithfulness_judge_protocol(self, settings):
        from src.evaluation.faithfulness import FaithfulnessJudgeProtocol
        from src.evaluation.providers.faithfulness_judge_anthropic import (
            AnthropicFaithfulnessJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicFaithfulnessJudge(settings)

        assert isinstance(judge, FaithfulnessJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.evaluation.providers.faithfulness_judge_anthropic import (
            AnthropicFaithfulnessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            AnthropicFaithfulnessJudge(settings)

        MockAnthropic.assert_called_once_with(api_key=settings.anthropic_api_key)

    def test_anthropic_imported_lazily(self):
        import sys

        anthropic_mod = sys.modules.pop("anthropic", None)
        try:
            sys.modules.pop(
                "src.evaluation.providers.faithfulness_judge_anthropic", None
            )
            import src.evaluation.providers.faithfulness_judge_anthropic  # noqa: F401

            assert "anthropic" not in sys.modules
        finally:
            if anthropic_mod is not None:
                sys.modules["anthropic"] = anthropic_mod

    def test_default_model_constant(self):
        from src.evaluation.providers.faithfulness_judge_anthropic import DEFAULT_MODEL

        assert DEFAULT_MODEL.startswith("claude-")

    def test_judge_raises_runtime_error_when_parsed_output_is_none(self, settings):
        from src.evaluation.providers.faithfulness_judge_anthropic import (
            AnthropicFaithfulnessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            resp = MagicMock()
            resp.parsed_output = None
            MockAnthropic.return_value.messages.parse.return_value = resp
            judge = AnthropicFaithfulnessJudge(settings)

            with pytest.raises(
                RuntimeError,
                match="Anthropic structured output returned no parsed_output",
            ):
                judge.judge(claim="c", context="e")

    def test_judge_records_span_with_prompt_and_token_count(self, settings):
        from src.evaluation.providers.faithfulness_judge_anthropic import (
            AnthropicFaithfulnessJudge,
        )
        from src.tracing.context import collect_spans

        resp = _mock_response(grounded=True, reasoning="Matches.")
        resp.usage.input_tokens = 100
        resp.usage.output_tokens = 20

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = resp
            judge = AnthropicFaithfulnessJudge(settings)
            with collect_spans() as spans:
                judge.judge(claim="The sky is blue.", context="The sky appears blue.")

        assert len(spans) == 1
        assert spans[0].step == "verification"
        assert spans[0].token_count == 120
        assert spans[0].error is None
