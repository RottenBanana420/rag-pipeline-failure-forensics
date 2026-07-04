"""Unit tests for AnthropicCitationJudge — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_response(
    supported: bool = True, reasoning: str = "Evidence backs it up."
) -> MagicMock:
    from src.generation.citation_verifier import JudgeVerdict

    resp = MagicMock()
    resp.parsed_output = JudgeVerdict(supported=supported, reasoning=reasoning)
    return resp


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestAnthropicCitationJudge:
    def test_importable(self):
        from src.generation.providers.citation_judge_anthropic import (  # noqa: F401
            AnthropicCitationJudge,
        )

    def test_judge_returns_judge_verdict(self, settings):
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response(
                supported=True, reasoning="Matches the evidence."
            )
            judge = AnthropicCitationJudge(settings)
            verdict = judge.judge(
                claim="The sky is blue.", evidence="The sky appears blue."
            )

        assert verdict.supported is True
        assert verdict.reasoning == "Matches the evidence."

    def test_judge_maps_unsupported_verdict(self, settings):
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response(
                supported=False, reasoning="Evidence is unrelated."
            )
            judge = AnthropicCitationJudge(settings)
            verdict = judge.judge(claim="Claim", evidence="Unrelated evidence")

        assert verdict.supported is False
        assert verdict.reasoning == "Evidence is unrelated."

    def test_judge_calls_sdk_with_correct_model_system_temperature(self, settings):
        from src.generation.citation_verifier import CITATION_JUDGE_SYSTEM_PROMPT
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicCitationJudge(settings)
            judge.judge(claim="The sky is blue.", evidence="The sky appears blue.")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.citation_judge_model
        assert kwargs["system"] == CITATION_JUDGE_SYSTEM_PROMPT
        assert kwargs["temperature"] == settings.citation_judge_temperature

    def test_judge_builds_prompt_via_build_judge_prompt(self, settings):
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicCitationJudge(settings)
            judge.judge(claim="The sky is blue.", evidence="The sky appears blue.")

        kwargs = mock_parse.call_args.kwargs
        messages = kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "The sky is blue." in messages[0]["content"]
        assert "The sky appears blue." in messages[0]["content"]
        # build_judge_prompt wraps claim/evidence in nonce-suffixed tags.
        assert "<claim-" in messages[0]["content"]
        assert "<evidence-" in messages[0]["content"]

    def test_judge_passes_output_format_as_judge_verdict(self, settings):
        from src.generation.citation_verifier import JudgeVerdict
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            mock_parse.return_value = _mock_response()
            judge = AnthropicCitationJudge(settings)
            judge.judge(claim="claim", evidence="evidence")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["output_format"] is JudgeVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicCitationJudge(settings)

        assert judge.provider_id == f"anthropic/{settings.citation_judge_model}"

    def test_satisfies_citation_judge_protocol(self, settings):
        from src.generation.citation_verifier import CitationJudgeProtocol
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )

        with patch("anthropic.Anthropic"):
            judge = AnthropicCitationJudge(settings)

        assert isinstance(judge, CitationJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            AnthropicCitationJudge(settings)

        MockAnthropic.assert_called_once_with(api_key=settings.anthropic_api_key)

    def test_anthropic_imported_lazily(self):
        """anthropic should not be imported at module top-level in the provider file."""
        import sys

        anthropic_mod = sys.modules.pop("anthropic", None)
        try:
            if "src.generation.providers.citation_judge_anthropic" in sys.modules:
                del sys.modules["src.generation.providers.citation_judge_anthropic"]
            import src.generation.providers.citation_judge_anthropic  # noqa: F401

            assert "anthropic" not in sys.modules
        finally:
            if anthropic_mod is not None:
                sys.modules["anthropic"] = anthropic_mod

    def test_default_model_constant(self):
        from src.generation.providers.citation_judge_anthropic import DEFAULT_MODEL

        assert isinstance(DEFAULT_MODEL, str)
        assert DEFAULT_MODEL.startswith("claude-")

    def test_judge_raises_runtime_error_when_parsed_output_is_none(self, settings):
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_parse = MockAnthropic.return_value.messages.parse
            resp = MagicMock()
            resp.parsed_output = None
            mock_parse.return_value = resp
            judge = AnthropicCitationJudge(settings)

            with pytest.raises(
                RuntimeError,
                match="Anthropic structured output returned no parsed_output",
            ):
                judge.judge(claim="claim", evidence="evidence")

    def test_judge_records_span_with_prompt_and_token_count(self, settings):
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )
        from src.tracing.context import collect_spans

        resp = _mock_response(supported=True, reasoning="Matches.")
        resp.usage.input_tokens = 120
        resp.usage.output_tokens = 30

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = resp
            judge = AnthropicCitationJudge(settings)
            with collect_spans() as spans:
                judge.judge(claim="The sky is blue.", evidence="The sky appears blue.")

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "verification"
        assert recorded.token_count == 150
        assert "The sky is blue." in recorded.llm_prompt
        assert recorded.error is None

    def test_judge_noop_outside_collect_spans(self, settings):
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response()
            judge = AnthropicCitationJudge(settings)
            judge.judge(claim="c", evidence="e")
