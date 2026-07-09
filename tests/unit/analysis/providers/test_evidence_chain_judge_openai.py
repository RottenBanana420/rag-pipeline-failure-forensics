"""Unit tests for OpenAIEvidenceChainJudge — TDD (written before implementation)."""

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


def _mock_completion(
    narrative: str = "Retrieval ranked poorly, which propagated to Generation.",
) -> MagicMock:
    from src.analysis.evidence_chain import EvidenceChainVerdict

    completion = MagicMock()
    completion.choices[0].message.parsed = EvidenceChainVerdict(narrative=narrative)
    completion.choices[0].message.refusal = None
    return completion


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestOpenAIEvidenceChainJudge:
    def test_importable(self):
        from src.analysis.providers.evidence_chain_judge_openai import (  # noqa: F401
            OpenAIEvidenceChainJudge,
        )

    def test_narrate_returns_evidence_chain_verdict(self, settings):
        from src.analysis.providers.evidence_chain_judge_openai import (
            OpenAIEvidenceChainJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion(
                    narrative="Retrieval ranked the chunk at position 7 instead of 1."
                )
            )
            judge = OpenAIEvidenceChainJudge(settings)
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
        from src.analysis.providers.evidence_chain_judge_openai import (
            OpenAIEvidenceChainJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIEvidenceChainJudge(settings)
            entries = _entries()
            judge.narrate(
                category="retrieval_failure", category_rationale="rat", chain=entries
            )

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.evidence_chain_judge_model
        assert kwargs["temperature"] == settings.evidence_chain_judge_temperature
        messages = kwargs["messages"]
        assert messages[0]["role"] == "system"
        expected_system = build_evidence_chain_judge_prompt(
            "retrieval_failure", "rat", entries
        ).system
        assert messages[0]["content"] == expected_system

    def test_narrate_builds_prompt_via_build_evidence_chain_judge_prompt(
        self, settings
    ):
        from src.analysis.providers.evidence_chain_judge_openai import (
            OpenAIEvidenceChainJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIEvidenceChainJudge(settings)
            judge.narrate(
                category="context_loss",
                category_rationale="the rationale",
                chain=_entries(),
            )

        kwargs = mock_parse.call_args.kwargs
        messages = kwargs["messages"]
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert "the query" in messages[1]["content"]
        assert "the chunks" in messages[1]["content"]
        assert "the rationale" in messages[1]["content"]
        assert "<span-0-input-" in messages[1]["content"]
        assert "<span-1-output-" in messages[1]["content"]
        assert "<category-rationale-" in messages[1]["content"]

    def test_narrate_passes_response_format_as_evidence_chain_verdict(self, settings):
        from src.analysis.evidence_chain import EvidenceChainVerdict
        from src.analysis.providers.evidence_chain_judge_openai import (
            OpenAIEvidenceChainJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAIEvidenceChainJudge(settings)
            judge.narrate(
                category="retrieval_failure", category_rationale="rat", chain=_entries()
            )

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["response_format"] is EvidenceChainVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.analysis.providers.evidence_chain_judge_openai import (
            OpenAIEvidenceChainJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAIEvidenceChainJudge(settings)

        assert judge.provider_id == f"openai/{settings.evidence_chain_judge_model}"

    def test_satisfies_evidence_chain_judge_protocol(self, settings):
        from src.analysis.evidence_chain import EvidenceChainJudgeProtocol
        from src.analysis.providers.evidence_chain_judge_openai import (
            OpenAIEvidenceChainJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAIEvidenceChainJudge(settings)

        assert isinstance(judge, EvidenceChainJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.analysis.providers.evidence_chain_judge_openai import (
            OpenAIEvidenceChainJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            OpenAIEvidenceChainJudge(settings)

        MockOpenAI.assert_called_once_with(api_key=settings.openai_api_key)

    def test_openai_imported_lazily(self):
        import sys

        openai_mod = sys.modules.pop("openai", None)
        try:
            if "src.analysis.providers.evidence_chain_judge_openai" in sys.modules:
                del sys.modules["src.analysis.providers.evidence_chain_judge_openai"]
            import src.analysis.providers.evidence_chain_judge_openai  # noqa: F401

            assert "openai" not in sys.modules
        finally:
            if openai_mod is not None:
                sys.modules["openai"] = openai_mod

    def test_default_model_constant(self):
        from src.analysis.providers.evidence_chain_judge_openai import DEFAULT_MODEL

        assert isinstance(DEFAULT_MODEL, str)
        assert DEFAULT_MODEL.startswith("gpt-")

    def test_narrate_raises_runtime_error_when_parsed_is_none(self, settings):
        from src.analysis.providers.evidence_chain_judge_openai import (
            OpenAIEvidenceChainJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            completion = MagicMock()
            completion.choices[0].message.parsed = None
            completion.choices[0].message.refusal = "I cannot assess this."
            mock_parse.return_value = completion
            judge = OpenAIEvidenceChainJudge(settings)

            with pytest.raises(
                RuntimeError,
                match="OpenAI structured output returned no parsed result",
            ):
                judge.narrate(
                    category="retrieval_failure",
                    category_rationale="rat",
                    chain=_entries(),
                )

    def test_narrate_records_span_with_step_analysis_prompt_and_token_count(
        self, settings
    ):
        from src.analysis.providers.evidence_chain_judge_openai import (
            OpenAIEvidenceChainJudge,
        )
        from src.tracing.context import collect_spans

        completion = _mock_completion(narrative="Narrative text.")
        completion.usage.total_tokens = 300

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = completion
            judge = OpenAIEvidenceChainJudge(settings)
            with collect_spans() as spans:
                judge.narrate(
                    category="context_loss",
                    category_rationale="rat",
                    chain=_entries(),
                )

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "analysis"
        assert recorded.token_count == 300
        assert "the query" in recorded.llm_prompt
        assert recorded.error is None

    def test_narrate_noop_outside_collect_spans(self, settings):
        from src.analysis.providers.evidence_chain_judge_openai import (
            OpenAIEvidenceChainJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion()
            )
            judge = OpenAIEvidenceChainJudge(settings)
            judge.narrate(
                category="retrieval_failure", category_rationale="rat", chain=_entries()
            )
