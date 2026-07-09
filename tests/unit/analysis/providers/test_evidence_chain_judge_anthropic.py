"""Unit tests for AnthropicEvidenceChainJudge — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _entries():
    from src.analysis.evidence_chain import EvidenceEntry

    return [
        EvidenceEntry(
            step="retrieval",
            input="the query",
            output="the chunks",
            score=1,
            rationale="rat0",
        ),
        EvidenceEntry(
            step="generation",
            input="prompt",
            output="answer",
            score=2,
            rationale="rat1",
        ),
    ]


def _mock_response(
    narrative: str = "Retrieval ranked poorly, which propagated to Generation.",
) -> MagicMock:
    from src.analysis.evidence_chain import EvidenceChainVerdict

    resp = MagicMock()
    resp.parsed_output = EvidenceChainVerdict(narrative=narrative)
    return resp


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestAnthropicEvidenceChainJudge:
    def test_importable(self):
        from src.analysis.providers.evidence_chain_judge_anthropic import (  # noqa: F401
            AnthropicEvidenceChainJudge,
        )

    def test_narrate_returns_evidence_chain_verdict(self, settings):
        from src.analysis.providers.evidence_chain_judge_anthropic import (
            AnthropicEvidenceChainJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response(
                narrative="Retrieval ranked the chunk at position 7 instead of 1."
            )
            judge = AnthropicEvidenceChainJudge(settings)
            verdict = judge.narrate(
                category="retrieval_failure",
                category_rationale="wrong docs retrieved",
                chain=_entries(),
            )

        assert (
            verdict.narrative
            == "Retrieval ranked the chunk at position 7 instead of 1."
        )

    def test_narrate_calls_sdk_with_correct_model_temperature(self, settings):
        from src.analysis.evidence_chain import build_evidence_chain_judge_prompt
        from src.analysis.providers.evidence_chain_judge_anthropic import (
            AnthropicEvidenceChainJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicEvidenceChainJudge(settings)
            entries = _entries()
            judge.narrate(
                category="retrieval_failure", category_rationale="rat", chain=entries
            )

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.evidence_chain_judge_model
        assert kwargs["temperature"] == settings.evidence_chain_judge_temperature
        expected_system = build_evidence_chain_judge_prompt(
            "retrieval_failure", "rat", entries
        ).system
        assert kwargs["system"] == expected_system

    def test_narrate_builds_prompt_via_build_evidence_chain_judge_prompt(
        self, settings
    ):
        from src.analysis.providers.evidence_chain_judge_anthropic import (
            AnthropicEvidenceChainJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicEvidenceChainJudge(settings)
            judge.narrate(
                category="context_loss",
                category_rationale="the rationale",
                chain=_entries(),
            )

        kwargs = mock_parse.call_args.kwargs
        messages = kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "the query" in messages[0]["content"]
        assert "the chunks" in messages[0]["content"]
        assert "the rationale" in messages[0]["content"]
        assert "<span-0-input-" in messages[0]["content"]
        assert "<span-1-output-" in messages[0]["content"]
        assert "<category-rationale-" in messages[0]["content"]

    def test_narrate_passes_output_format_as_evidence_chain_verdict(self, settings):
        from src.analysis.evidence_chain import EvidenceChainVerdict
        from src.analysis.providers.evidence_chain_judge_anthropic import (
            AnthropicEvidenceChainJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicEvidenceChainJudge(settings)
            judge.narrate(
                category="retrieval_failure", category_rationale="rat", chain=_entries()
            )

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["output_format"] is EvidenceChainVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.analysis.providers.evidence_chain_judge_anthropic import (
            AnthropicEvidenceChainJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicEvidenceChainJudge(settings)

        assert judge.provider_id == f"anthropic/{settings.evidence_chain_judge_model}"

    def test_satisfies_evidence_chain_judge_protocol(self, settings):
        from src.analysis.evidence_chain import EvidenceChainJudgeProtocol
        from src.analysis.providers.evidence_chain_judge_anthropic import (
            AnthropicEvidenceChainJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicEvidenceChainJudge(settings)

        assert isinstance(judge, EvidenceChainJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.analysis.providers.evidence_chain_judge_anthropic import (
            AnthropicEvidenceChainJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            AnthropicEvidenceChainJudge(settings)

        MockAnthropic.assert_called_once_with(api_key=settings.anthropic_api_key)

    def test_anthropic_imported_lazily(self):
        import sys

        anthropic_mod = sys.modules.pop("anthropic", None)
        try:
            if "src.analysis.providers.evidence_chain_judge_anthropic" in sys.modules:
                del sys.modules["src.analysis.providers.evidence_chain_judge_anthropic"]
            import src.analysis.providers.evidence_chain_judge_anthropic  # noqa: F401

            assert "anthropic" not in sys.modules
        finally:
            if anthropic_mod is not None:
                sys.modules["anthropic"] = anthropic_mod

    def test_default_model_constant(self):
        from src.analysis.providers.evidence_chain_judge_anthropic import DEFAULT_MODEL

        assert isinstance(DEFAULT_MODEL, str)
        assert DEFAULT_MODEL.startswith("claude-")

    def test_narrate_raises_runtime_error_when_parsed_output_is_none(self, settings):
        from src.analysis.providers.evidence_chain_judge_anthropic import (
            AnthropicEvidenceChainJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            resp = MagicMock()
            resp.parsed_output = None
            mock_parse.return_value = resp
            judge = AnthropicEvidenceChainJudge(settings)

            with pytest.raises(
                RuntimeError,
                match="Anthropic structured output returned no parsed_output",
            ):
                judge.narrate(
                    category="retrieval_failure",
                    category_rationale="rat",
                    chain=_entries(),
                )

    def test_narrate_records_span_with_step_analysis_prompt_and_token_count(
        self, settings
    ):
        from src.analysis.providers.evidence_chain_judge_anthropic import (
            AnthropicEvidenceChainJudge,
        )
        from src.tracing.context import collect_spans

        resp = _mock_response(narrative="Narrative text.")
        resp.usage.input_tokens = 200
        resp.usage.output_tokens = 50

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = resp
            judge = AnthropicEvidenceChainJudge(settings)
            with collect_spans() as spans:
                judge.narrate(
                    category="context_loss",
                    category_rationale="rat",
                    chain=_entries(),
                )

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "analysis"
        assert recorded.token_count == 250
        assert "the query" in recorded.llm_prompt
        assert recorded.error is None

    def test_narrate_noop_outside_collect_spans(self, settings):
        from src.analysis.providers.evidence_chain_judge_anthropic import (
            AnthropicEvidenceChainJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response()
            judge = AnthropicEvidenceChainJudge(settings)
            judge.narrate(
                category="retrieval_failure", category_rationale="rat", chain=_entries()
            )
