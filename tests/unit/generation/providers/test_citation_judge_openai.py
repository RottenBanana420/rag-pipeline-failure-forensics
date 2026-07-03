"""Unit tests for OpenAICitationJudge — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_completion(
    supported: bool = True, reasoning: str = "Evidence backs it up."
) -> MagicMock:
    from src.generation.citation_verifier import JudgeVerdict

    completion = MagicMock()
    completion.choices[0].message.parsed = JudgeVerdict(
        supported=supported, reasoning=reasoning
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


class TestOpenAICitationJudge:
    def test_importable(self):
        from src.generation.providers.citation_judge_openai import (  # noqa: F401
            OpenAICitationJudge,
        )

    def test_judge_returns_judge_verdict(self, settings):
        from src.generation.providers.citation_judge_openai import (
            OpenAICitationJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion(supported=True, reasoning="Matches the evidence.")
            )
            judge = OpenAICitationJudge(settings)
            verdict = judge.judge(
                claim="The sky is blue.", evidence="The sky appears blue."
            )

        assert verdict.supported is True
        assert verdict.reasoning == "Matches the evidence."

    def test_judge_maps_unsupported_verdict(self, settings):
        from src.generation.providers.citation_judge_openai import (
            OpenAICitationJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion(supported=False, reasoning="Evidence is unrelated.")
            )
            judge = OpenAICitationJudge(settings)
            verdict = judge.judge(claim="Claim", evidence="Unrelated evidence")

        assert verdict.supported is False
        assert verdict.reasoning == "Evidence is unrelated."

    def test_judge_calls_sdk_with_correct_model_system_temperature(self, settings):
        from src.generation.citation_verifier import CITATION_JUDGE_SYSTEM_PROMPT
        from src.generation.providers.citation_judge_openai import (
            OpenAICitationJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAICitationJudge(settings)
            judge.judge(claim="The sky is blue.", evidence="The sky appears blue.")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["model"] == settings.citation_judge_model
        assert kwargs["temperature"] == settings.citation_judge_temperature
        messages = kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == CITATION_JUDGE_SYSTEM_PROMPT

    def test_judge_builds_prompt_via_build_judge_prompt(self, settings):
        from src.generation.providers.citation_judge_openai import (
            OpenAICitationJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAICitationJudge(settings)
            judge.judge(claim="The sky is blue.", evidence="The sky appears blue.")

        kwargs = mock_parse.call_args.kwargs
        messages = kwargs["messages"]
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert "The sky is blue." in messages[1]["content"]
        assert "The sky appears blue." in messages[1]["content"]
        # build_judge_prompt wraps claim/evidence in nonce-suffixed tags.
        assert "<claim-" in messages[1]["content"]
        assert "<evidence-" in messages[1]["content"]

    def test_judge_passes_response_format_as_judge_verdict(self, settings):
        from src.generation.citation_verifier import JudgeVerdict
        from src.generation.providers.citation_judge_openai import (
            OpenAICitationJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            mock_parse.return_value = _mock_completion()
            judge = OpenAICitationJudge(settings)
            judge.judge(claim="claim", evidence="evidence")

        kwargs = mock_parse.call_args.kwargs
        assert kwargs["response_format"] is JudgeVerdict

    def test_provider_id_includes_model_name(self, settings):
        from src.generation.providers.citation_judge_openai import (
            OpenAICitationJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAICitationJudge(settings)

        assert judge.provider_id == f"openai/{settings.citation_judge_model}"

    def test_satisfies_citation_judge_protocol(self, settings):
        from src.generation.citation_verifier import CitationJudgeProtocol
        from src.generation.providers.citation_judge_openai import (
            OpenAICitationJudge,
        )

        with patch("openai.OpenAI"):
            judge = OpenAICitationJudge(settings)

        assert isinstance(judge, CitationJudgeProtocol)

    def test_client_constructed_with_api_key_from_settings(self, settings):
        from src.generation.providers.citation_judge_openai import (
            OpenAICitationJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            OpenAICitationJudge(settings)

        MockOpenAI.assert_called_once_with(api_key=settings.openai_api_key)

    def test_openai_imported_lazily(self):
        """openai should not be imported at module top-level in the provider file."""
        import sys

        openai_mod = sys.modules.pop("openai", None)
        try:
            if "src.generation.providers.citation_judge_openai" in sys.modules:
                del sys.modules["src.generation.providers.citation_judge_openai"]
            import src.generation.providers.citation_judge_openai  # noqa: F401

            assert "openai" not in sys.modules
        finally:
            if openai_mod is not None:
                sys.modules["openai"] = openai_mod

    def test_default_model_constant(self):
        from src.generation.providers.citation_judge_openai import DEFAULT_MODEL

        assert isinstance(DEFAULT_MODEL, str)
        assert DEFAULT_MODEL.startswith("gpt-")

    def test_judge_raises_runtime_error_when_parsed_is_none(self, settings):
        from src.generation.providers.citation_judge_openai import (
            OpenAICitationJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            mock_parse = MockOpenAI.return_value.chat.completions.parse
            completion = MagicMock()
            completion.choices[0].message.parsed = None
            completion.choices[0].message.refusal = "I cannot assess this."
            mock_parse.return_value = completion
            judge = OpenAICitationJudge(settings)

            with pytest.raises(
                RuntimeError,
                match="OpenAI structured output returned no parsed result",
            ):
                judge.judge(claim="claim", evidence="evidence")
