"""Unit tests for OpenAIStepQualityJudge — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_completion(
    score: int = 4, rationale: str = "Reasonable transformation."
) -> MagicMock:
    from src.analysis.root_cause import StepQualityVerdict

    completion = MagicMock()
    completion.choices[0].message.parsed = StepQualityVerdict(
        score=score, rationale=rationale
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


class TestOpenAIStepQualityJudge:
    def test_importable(self):
        from src.analysis.providers.step_quality_judge_openai import (  # noqa: F401
            OpenAIStepQualityJudge,
        )

    def test_judge_returns_step_quality_verdict(self, settings):
        from src.analysis.providers.step_quality_judge_openai import (
            OpenAIStepQualityJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion(score=4, rationale="Chunks are topically relevant.")
            )
            judge = OpenAIStepQualityJudge(settings)
            verdict = judge.judge(
                step="retrieval", input="query embedding", output="retrieved chunks"
            )

        assert verdict.score == 4
        assert verdict.rationale == "Chunks are topically relevant."

    def test_judge_maps_low_score_verdict(self, settings):
        from src.analysis.providers.step_quality_judge_openai import (
            OpenAIStepQualityJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion(score=1, rationale="Completely unrelated chunks.")
            )
            judge = OpenAIStepQualityJudge(settings)
            verdict = judge.judge(step="retrieval", input="q", output="junk")

        assert verdict.score == 1
        assert verdict.rationale == "Completely unrelated chunks."

    def test_judge_calls_sdk_with_correct_model_temperature(self, settings):
        from src.analysis.providers.step_quality_judge_openai import (
            OpenAIStepQualityJudge,
        )
        from src.analysis.root_cause import build_step_quality_judge_prompt

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIStepQualityJudge(settings)
            judge.judge(step="ranking", input="in", output="out")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.root_cause_judge_model
        assert kwargs["temperature"] == settings.root_cause_judge_temperature
        messages = kwargs["messages"]
        assert messages[0]["role"] == "system"
        expected_system = build_step_quality_judge_prompt("ranking", "in", "out").system
        assert messages[0]["content"] == expected_system

    def test_judge_system_prompt_varies_by_step(self, settings):
        from src.analysis.providers.step_quality_judge_openai import (
            OpenAIStepQualityJudge,
        )
        from src.analysis.root_cause import STEP_QUALITY_CRITERIA

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIStepQualityJudge(settings)
            judge.judge(step="retrieval", input="in", output="out")
            retrieval_system = mock_parse.call_args.kwargs["messages"][0]["content"]
            judge.judge(step="generation", input="in", output="out")
            generation_system = mock_parse.call_args.kwargs["messages"][0]["content"]

        assert retrieval_system != generation_system
        assert STEP_QUALITY_CRITERIA["retrieval"] in retrieval_system
        assert STEP_QUALITY_CRITERIA["generation"] in generation_system

    def test_judge_builds_prompt_via_build_step_quality_judge_prompt(self, settings):
        from src.analysis.providers.step_quality_judge_openai import (
            OpenAIStepQualityJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIStepQualityJudge(settings)
            judge.judge(step="retrieval", input="the query", output="the chunks")

        kwargs = mock_parse.call_args.kwargs
        messages = kwargs["messages"]
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert "the query" in messages[1]["content"]
        assert "the chunks" in messages[1]["content"]
        assert "<input-" in messages[1]["content"]
        assert "<output-" in messages[1]["content"]

    def test_judge_passes_response_format_as_step_quality_verdict(self, settings):
        from src.analysis.providers.step_quality_judge_openai import (
            OpenAIStepQualityJudge,
        )
        from src.analysis.root_cause import StepQualityVerdict

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIStepQualityJudge(settings)
            judge.judge(step="retrieval", input="in", output="out")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["response_format"] is StepQualityVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.analysis.providers.step_quality_judge_openai import (
            OpenAIStepQualityJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAIStepQualityJudge(settings)

        assert judge.provider_id == f"openai/{settings.root_cause_judge_model}"

    def test_satisfies_step_quality_judge_protocol(self, settings):
        from src.analysis.providers.step_quality_judge_openai import (
            OpenAIStepQualityJudge,
        )
        from src.analysis.root_cause import StepQualityJudgeProtocol

        with patch("openai.OpenAI"):
            judge = OpenAIStepQualityJudge(settings)

        assert isinstance(judge, StepQualityJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.analysis.providers.step_quality_judge_openai import (
            OpenAIStepQualityJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            OpenAIStepQualityJudge(settings)

        MockOpenAI.assert_called_once_with(api_key=settings.openai_api_key)

    def test_openai_imported_lazily(self):
        """openai should not be imported at module top-level in the provider file."""
        import sys

        openai_mod = sys.modules.pop("openai", None)
        try:
            if "src.analysis.providers.step_quality_judge_openai" in sys.modules:
                del sys.modules["src.analysis.providers.step_quality_judge_openai"]
            import src.analysis.providers.step_quality_judge_openai  # noqa: F401

            assert "openai" not in sys.modules
        finally:
            if openai_mod is not None:
                sys.modules["openai"] = openai_mod

    def test_default_model_constant(self):
        from src.analysis.providers.step_quality_judge_openai import DEFAULT_MODEL

        assert isinstance(DEFAULT_MODEL, str)
        assert DEFAULT_MODEL.startswith("gpt-")

    def test_judge_raises_runtime_error_when_parsed_is_none(self, settings):
        from src.analysis.providers.step_quality_judge_openai import (
            OpenAIStepQualityJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            completion = MagicMock()
            completion.choices[0].message.parsed = None
            completion.choices[0].message.refusal = "I cannot assess this."
            mock_parse.return_value = completion
            judge = OpenAIStepQualityJudge(settings)

            with pytest.raises(
                RuntimeError,
                match="OpenAI structured output returned no parsed result",
            ):
                judge.judge(step="retrieval", input="in", output="out")

    def test_judge_records_span_with_step_analysis_prompt_and_token_count(
        self, settings
    ):
        from src.analysis.providers.step_quality_judge_openai import (
            OpenAIStepQualityJudge,
        )
        from src.tracing.context import collect_spans

        completion = _mock_completion(score=4, rationale="Reasonable.")
        completion.usage.total_tokens = 150

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = completion
            judge = OpenAIStepQualityJudge(settings)
            with collect_spans() as spans:
                judge.judge(
                    step="generation", input="a grounded prompt", output="an answer"
                )

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "analysis"
        assert recorded.token_count == 150
        assert "a grounded prompt" in recorded.llm_prompt
        assert recorded.error is None

    def test_judge_noop_outside_collect_spans(self, settings):
        from src.analysis.providers.step_quality_judge_openai import (
            OpenAIStepQualityJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion()
            )
            judge = OpenAIStepQualityJudge(settings)
            judge.judge(step="retrieval", input="in", output="out")
