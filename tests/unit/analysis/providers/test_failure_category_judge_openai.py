"""Unit tests for OpenAIFailureCategoryJudge — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_completion(
    category: str = "retrieval_failure", rationale: str = "No relevant chunks."
) -> MagicMock:
    from src.analysis.failure_categorizer import FailureCategoryVerdict

    completion = MagicMock()
    completion.choices[0].message.parsed = FailureCategoryVerdict(
        category=category, rationale=rationale
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


class TestOpenAIFailureCategoryJudge:
    def test_importable(self):
        from src.analysis.providers.failure_category_judge_openai import (  # noqa: F401
            OpenAIFailureCategoryJudge,
        )

    def test_classify_returns_failure_category_verdict(self, settings):
        from src.analysis.providers.failure_category_judge_openai import (
            OpenAIFailureCategoryJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion(
                    category="ranking_failure",
                    rationale="Reranker demoted the right chunk.",
                )
            )
            judge = OpenAIFailureCategoryJudge(settings)
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
        from src.analysis.providers.failure_category_judge_openai import (
            OpenAIFailureCategoryJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIFailureCategoryJudge(settings)
            judge.classify(
                step="ranking", input="in", output="out", quality_rationale="rat"
            )

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.failure_category_judge_model
        assert kwargs["temperature"] == settings.failure_category_judge_temperature
        messages = kwargs["messages"]
        assert messages[0]["role"] == "system"
        expected_system = build_failure_category_judge_prompt(
            "ranking", "in", "out", "rat"
        ).system
        assert messages[0]["content"] == expected_system

    def test_classify_builds_prompt_via_build_failure_category_judge_prompt(
        self, settings
    ):
        from src.analysis.providers.failure_category_judge_openai import (
            OpenAIFailureCategoryJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIFailureCategoryJudge(settings)
            judge.classify(
                step="retrieval",
                input="the query",
                output="the chunks",
                quality_rationale="the rationale",
            )

        kwargs = mock_parse.call_args.kwargs
        messages = kwargs["messages"]
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert "the query" in messages[1]["content"]
        assert "the chunks" in messages[1]["content"]
        assert "the rationale" in messages[1]["content"]
        assert "<input-" in messages[1]["content"]
        assert "<output-" in messages[1]["content"]
        assert "<quality-rationale-" in messages[1]["content"]

    def test_classify_passes_response_format_as_failure_category_verdict(
        self, settings
    ):
        from src.analysis.failure_categorizer import FailureCategoryVerdict
        from src.analysis.providers.failure_category_judge_openai import (
            OpenAIFailureCategoryJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIFailureCategoryJudge(settings)
            judge.classify(
                step="retrieval", input="in", output="out", quality_rationale="rat"
            )

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["response_format"] is FailureCategoryVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.analysis.providers.failure_category_judge_openai import (
            OpenAIFailureCategoryJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAIFailureCategoryJudge(settings)

        assert judge.provider_id == f"openai/{settings.failure_category_judge_model}"

    def test_satisfies_failure_category_judge_protocol(self, settings):
        from src.analysis.failure_categorizer import FailureCategoryJudgeProtocol
        from src.analysis.providers.failure_category_judge_openai import (
            OpenAIFailureCategoryJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAIFailureCategoryJudge(settings)

        assert isinstance(judge, FailureCategoryJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.analysis.providers.failure_category_judge_openai import (
            OpenAIFailureCategoryJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            OpenAIFailureCategoryJudge(settings)

        MockOpenAI.assert_called_once_with(api_key=settings.openai_api_key)

    def test_openai_imported_lazily(self):
        import sys

        openai_mod = sys.modules.pop("openai", None)
        try:
            if "src.analysis.providers.failure_category_judge_openai" in sys.modules:
                del sys.modules["src.analysis.providers.failure_category_judge_openai"]
            import src.analysis.providers.failure_category_judge_openai  # noqa: F401

            assert "openai" not in sys.modules
        finally:
            if openai_mod is not None:
                sys.modules["openai"] = openai_mod

    def test_default_model_constant(self):
        from src.analysis.providers.failure_category_judge_openai import DEFAULT_MODEL

        assert isinstance(DEFAULT_MODEL, str)
        assert DEFAULT_MODEL.startswith("gpt-")

    def test_classify_raises_runtime_error_when_parsed_is_none(self, settings):
        from src.analysis.providers.failure_category_judge_openai import (
            OpenAIFailureCategoryJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            completion = MagicMock()
            completion.choices[0].message.parsed = None
            completion.choices[0].message.refusal = "I cannot assess this."
            mock_parse.return_value = completion
            judge = OpenAIFailureCategoryJudge(settings)

            with pytest.raises(
                RuntimeError,
                match="OpenAI structured output returned no parsed result",
            ):
                judge.classify(
                    step="retrieval", input="in", output="out", quality_rationale="rat"
                )

    def test_classify_records_span_with_step_analysis_prompt_and_token_count(
        self, settings
    ):
        from src.analysis.providers.failure_category_judge_openai import (
            OpenAIFailureCategoryJudge,
        )
        from src.tracing.context import collect_spans

        completion = _mock_completion(
            category="citation_error", rationale="Unsupported claim."
        )
        completion.usage.total_tokens = 150

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = completion
            judge = OpenAIFailureCategoryJudge(settings)
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
        assert recorded.token_count == 150
        assert "a claim/evidence pair" in recorded.llm_prompt
        assert recorded.error is None

    def test_classify_noop_outside_collect_spans(self, settings):
        from src.analysis.providers.failure_category_judge_openai import (
            OpenAIFailureCategoryJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion()
            )
            judge = OpenAIFailureCategoryJudge(settings)
            judge.classify(
                step="retrieval", input="in", output="out", quality_rationale="rat"
            )
