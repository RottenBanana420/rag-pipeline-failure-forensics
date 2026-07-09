"""Unit tests for AnthropicFailureCategoryJudge — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_response(
    category: str = "retrieval_failure", rationale: str = "No relevant chunks."
) -> MagicMock:
    from src.analysis.failure_categorizer import FailureCategoryVerdict

    resp = MagicMock()
    resp.parsed_output = FailureCategoryVerdict(category=category, rationale=rationale)
    return resp


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestAnthropicFailureCategoryJudge:
    def test_importable(self):
        from src.analysis.providers.failure_category_judge_anthropic import (  # noqa: F401
            AnthropicFailureCategoryJudge,
        )

    def test_classify_returns_failure_category_verdict(self, settings):
        from src.analysis.providers.failure_category_judge_anthropic import (
            AnthropicFailureCategoryJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response(
                category="ranking_failure",
                rationale="Reranker demoted the right chunk.",
            )
            judge = AnthropicFailureCategoryJudge(settings)
            verdict = judge.classify(
                step="ranking",
                input="candidate pool",
                output="top-n",
                quality_rationale="bad",
            )

        assert verdict.category == "ranking_failure"
        assert verdict.rationale == "Reranker demoted the right chunk."

    def test_classify_calls_sdk_with_correct_model_temperature(self, settings):
        from src.analysis.failure_categorizer import build_failure_category_judge_prompt
        from src.analysis.providers.failure_category_judge_anthropic import (
            AnthropicFailureCategoryJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicFailureCategoryJudge(settings)
            judge.classify(
                step="ranking", input="in", output="out", quality_rationale="rat"
            )

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.failure_category_judge_model
        assert kwargs["temperature"] == settings.failure_category_judge_temperature
        expected_system = build_failure_category_judge_prompt(
            "ranking", "in", "out", "rat"
        ).system
        assert kwargs["system"] == expected_system

    def test_classify_builds_prompt_via_build_failure_category_judge_prompt(
        self, settings
    ):
        from src.analysis.providers.failure_category_judge_anthropic import (
            AnthropicFailureCategoryJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicFailureCategoryJudge(settings)
            judge.classify(
                step="retrieval",
                input="the query",
                output="the chunks",
                quality_rationale="the rationale",
            )

        kwargs = mock_parse.call_args.kwargs
        messages = kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "the query" in messages[0]["content"]
        assert "the chunks" in messages[0]["content"]
        assert "the rationale" in messages[0]["content"]
        assert "<input-" in messages[0]["content"]
        assert "<output-" in messages[0]["content"]
        assert "<quality-rationale-" in messages[0]["content"]

    def test_classify_passes_output_format_as_failure_category_verdict(self, settings):
        from src.analysis.failure_categorizer import FailureCategoryVerdict
        from src.analysis.providers.failure_category_judge_anthropic import (
            AnthropicFailureCategoryJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicFailureCategoryJudge(settings)
            judge.classify(
                step="retrieval", input="in", output="out", quality_rationale="rat"
            )

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["output_format"] is FailureCategoryVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.analysis.providers.failure_category_judge_anthropic import (
            AnthropicFailureCategoryJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicFailureCategoryJudge(settings)

        assert judge.provider_id == f"anthropic/{settings.failure_category_judge_model}"

    def test_satisfies_failure_category_judge_protocol(self, settings):
        from src.analysis.failure_categorizer import FailureCategoryJudgeProtocol
        from src.analysis.providers.failure_category_judge_anthropic import (
            AnthropicFailureCategoryJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicFailureCategoryJudge(settings)

        assert isinstance(judge, FailureCategoryJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.analysis.providers.failure_category_judge_anthropic import (
            AnthropicFailureCategoryJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            AnthropicFailureCategoryJudge(settings)

        MockAnthropic.assert_called_once_with(api_key=settings.anthropic_api_key)

    def test_anthropic_imported_lazily(self):
        import sys

        anthropic_mod = sys.modules.pop("anthropic", None)
        try:
            if "src.analysis.providers.failure_category_judge_anthropic" in sys.modules:
                del sys.modules[
                    "src.analysis.providers.failure_category_judge_anthropic"
                ]
            import src.analysis.providers.failure_category_judge_anthropic  # noqa: F401

            assert "anthropic" not in sys.modules
        finally:
            if anthropic_mod is not None:
                sys.modules["anthropic"] = anthropic_mod

    def test_default_model_constant(self):
        from src.analysis.providers.failure_category_judge_anthropic import (
            DEFAULT_MODEL,
        )

        assert isinstance(DEFAULT_MODEL, str)
        assert DEFAULT_MODEL.startswith("claude-")

    def test_classify_raises_runtime_error_when_parsed_output_is_none(self, settings):
        from src.analysis.providers.failure_category_judge_anthropic import (
            AnthropicFailureCategoryJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            resp = MagicMock()
            resp.parsed_output = None
            mock_parse.return_value = resp
            judge = AnthropicFailureCategoryJudge(settings)

            with pytest.raises(
                RuntimeError,
                match="Anthropic structured output returned no parsed_output",
            ):
                judge.classify(
                    step="retrieval", input="in", output="out", quality_rationale="rat"
                )

    def test_classify_records_span_with_step_analysis_prompt_and_token_count(
        self, settings
    ):
        from src.analysis.providers.failure_category_judge_anthropic import (
            AnthropicFailureCategoryJudge,
        )
        from src.tracing.context import collect_spans

        resp = _mock_response(category="citation_error", rationale="Unsupported claim.")
        resp.usage.input_tokens = 80
        resp.usage.output_tokens = 20

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = resp
            judge = AnthropicFailureCategoryJudge(settings)
            with collect_spans() as spans:
                judge.classify(
                    step="verification",
                    input="a claim/evidence pair",
                    output="an unsupported verdict",
                    quality_rationale="rat",
                )

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "analysis"
        assert recorded.token_count == 100
        assert "a claim/evidence pair" in recorded.llm_prompt
        assert recorded.error is None

    def test_classify_noop_outside_collect_spans(self, settings):
        from src.analysis.providers.failure_category_judge_anthropic import (
            AnthropicFailureCategoryJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response()
            judge = AnthropicFailureCategoryJudge(settings)
            judge.classify(
                step="retrieval", input="in", output="out", quality_rationale="rat"
            )
