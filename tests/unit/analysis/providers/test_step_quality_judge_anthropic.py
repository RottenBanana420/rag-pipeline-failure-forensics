"""Unit tests for AnthropicStepQualityJudge — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_response(
    score: int = 4, rationale: str = "Reasonable transformation."
) -> MagicMock:
    from src.analysis.root_cause import StepQualityVerdict

    resp = MagicMock()
    resp.parsed_output = StepQualityVerdict(score=score, rationale=rationale)
    return resp


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestAnthropicStepQualityJudge:
    def test_importable(self):
        from src.analysis.providers.step_quality_judge_anthropic import (  # noqa: F401
            AnthropicStepQualityJudge,
        )

    def test_judge_returns_step_quality_verdict(self, settings):
        from src.analysis.providers.step_quality_judge_anthropic import (
            AnthropicStepQualityJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response(
                score=4, rationale="Chunks are topically relevant."
            )
            judge = AnthropicStepQualityJudge(settings)
            verdict = judge.judge(
                step="retrieval", input="query embedding", output="retrieved chunks"
            )

        assert verdict.score == 4
        assert verdict.rationale == "Chunks are topically relevant."

    def test_judge_maps_low_score_verdict(self, settings):
        from src.analysis.providers.step_quality_judge_anthropic import (
            AnthropicStepQualityJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response(
                score=1, rationale="Completely unrelated chunks."
            )
            judge = AnthropicStepQualityJudge(settings)
            verdict = judge.judge(step="retrieval", input="q", output="junk")

        assert verdict.score == 1
        assert verdict.rationale == "Completely unrelated chunks."

    def test_judge_calls_sdk_with_correct_model_temperature(self, settings):
        from src.analysis.providers.step_quality_judge_anthropic import (
            AnthropicStepQualityJudge,
        )
        from src.analysis.root_cause import build_step_quality_judge_prompt

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicStepQualityJudge(settings)
            judge.judge(step="ranking", input="in", output="out")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.root_cause_judge_model
        assert kwargs["temperature"] == settings.root_cause_judge_temperature
        expected_system = build_step_quality_judge_prompt("ranking", "in", "out").system
        assert kwargs["system"] == expected_system

    def test_judge_system_prompt_varies_by_step(self, settings):
        from src.analysis.providers.step_quality_judge_anthropic import (
            AnthropicStepQualityJudge,
        )
        from src.analysis.root_cause import STEP_QUALITY_CRITERIA

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicStepQualityJudge(settings)
            judge.judge(step="retrieval", input="in", output="out")
            retrieval_system = mock_parse.call_args.kwargs["system"]
            judge.judge(step="generation", input="in", output="out")
            generation_system = mock_parse.call_args.kwargs["system"]

        assert retrieval_system != generation_system
        assert STEP_QUALITY_CRITERIA["retrieval"] in retrieval_system
        assert STEP_QUALITY_CRITERIA["generation"] in generation_system

    def test_judge_builds_prompt_via_build_step_quality_judge_prompt(self, settings):
        from src.analysis.providers.step_quality_judge_anthropic import (
            AnthropicStepQualityJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicStepQualityJudge(settings)
            judge.judge(step="retrieval", input="the query", output="the chunks")

        kwargs = mock_parse.call_args.kwargs
        messages = kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "the query" in messages[0]["content"]
        assert "the chunks" in messages[0]["content"]
        assert "<input-" in messages[0]["content"]
        assert "<output-" in messages[0]["content"]

    def test_judge_passes_output_format_as_step_quality_verdict(self, settings):
        from src.analysis.providers.step_quality_judge_anthropic import (
            AnthropicStepQualityJudge,
        )
        from src.analysis.root_cause import StepQualityVerdict

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicStepQualityJudge(settings)
            judge.judge(step="retrieval", input="in", output="out")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["output_format"] is StepQualityVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.analysis.providers.step_quality_judge_anthropic import (
            AnthropicStepQualityJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicStepQualityJudge(settings)

        assert judge.provider_id == f"anthropic/{settings.root_cause_judge_model}"

    def test_satisfies_step_quality_judge_protocol(self, settings):
        from src.analysis.providers.step_quality_judge_anthropic import (
            AnthropicStepQualityJudge,
        )
        from src.analysis.root_cause import StepQualityJudgeProtocol

        with patch("anthropic.Anthropic"):
            judge = AnthropicStepQualityJudge(settings)

        assert isinstance(judge, StepQualityJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.analysis.providers.step_quality_judge_anthropic import (
            AnthropicStepQualityJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            AnthropicStepQualityJudge(settings)

        MockAnthropic.assert_called_once_with(api_key=settings.anthropic_api_key)

    def test_anthropic_imported_lazily(self):
        """anthropic should not be imported at module top-level in the provider file."""
        import sys

        anthropic_mod = sys.modules.pop("anthropic", None)
        try:
            if "src.analysis.providers.step_quality_judge_anthropic" in sys.modules:
                del sys.modules["src.analysis.providers.step_quality_judge_anthropic"]
            import src.analysis.providers.step_quality_judge_anthropic  # noqa: F401

            assert "anthropic" not in sys.modules
        finally:
            if anthropic_mod is not None:
                sys.modules["anthropic"] = anthropic_mod

    def test_default_model_constant(self):
        from src.analysis.providers.step_quality_judge_anthropic import DEFAULT_MODEL

        assert isinstance(DEFAULT_MODEL, str)
        assert DEFAULT_MODEL.startswith("claude-")

    def test_judge_raises_runtime_error_when_parsed_output_is_none(self, settings):
        from src.analysis.providers.step_quality_judge_anthropic import (
            AnthropicStepQualityJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            resp = MagicMock()
            resp.parsed_output = None
            mock_parse.return_value = resp
            judge = AnthropicStepQualityJudge(settings)

            with pytest.raises(
                RuntimeError,
                match="Anthropic structured output returned no parsed_output",
            ):
                judge.judge(step="retrieval", input="in", output="out")

    def test_judge_records_span_with_step_analysis_prompt_and_token_count(
        self, settings
    ):
        from src.analysis.providers.step_quality_judge_anthropic import (
            AnthropicStepQualityJudge,
        )
        from src.tracing.context import collect_spans

        resp = _mock_response(score=4, rationale="Reasonable.")
        resp.usage.input_tokens = 80
        resp.usage.output_tokens = 20

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = resp
            judge = AnthropicStepQualityJudge(settings)
            with collect_spans() as spans:
                judge.judge(
                    step="generation", input="a grounded prompt", output="an answer"
                )

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "analysis"
        assert recorded.token_count == 100
        assert "a grounded prompt" in recorded.llm_prompt
        assert recorded.error is None

    def test_judge_noop_outside_collect_spans(self, settings):
        from src.analysis.providers.step_quality_judge_anthropic import (
            AnthropicStepQualityJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response()
            judge = AnthropicStepQualityJudge(settings)
            judge.judge(step="retrieval", input="in", output="out")
