"""Unit tests for src/evaluation/answer_correctness.py — mirrors test_citation_verifier.py's shape."""

import re
from unittest.mock import MagicMock, patch

import pytest

from src.generation.prompts import GroundedPrompt


class FakeJudge:
    def __init__(self, verdict=None, provider_id: str = "fake/v1") -> None:
        self._verdict = verdict
        self.calls: list[tuple[str, str, str]] = []
        self._provider_id = provider_id

    def judge(self, question: str, expected_answer: str, actual_answer: str):
        from src.evaluation.answer_correctness import CorrectnessVerdict

        self.calls.append((question, expected_answer, actual_answer))
        if self._verdict is not None:
            return self._verdict
        return CorrectnessVerdict(correct=True, reasoning="Default canned verdict.")

    @property
    def provider_id(self) -> str:
        return self._provider_id


class TestAnswerCorrectnessJudgeProtocol:
    def test_fake_judge_satisfies_protocol(self):
        from src.evaluation.answer_correctness import AnswerCorrectnessJudgeProtocol

        assert isinstance(FakeJudge(), AnswerCorrectnessJudgeProtocol)


class TestCorrectnessVerdict:
    def test_is_pydantic_model(self):
        from pydantic import BaseModel

        from src.evaluation.answer_correctness import CorrectnessVerdict

        assert issubclass(CorrectnessVerdict, BaseModel)

    def test_has_correct_and_reasoning_fields(self):
        from src.evaluation.answer_correctness import CorrectnessVerdict

        verdict = CorrectnessVerdict(correct=True, reasoning="Matches golden answer.")

        assert verdict.correct is True
        assert verdict.reasoning == "Matches golden answer."


class TestBuildAnswerCorrectnessJudgePrompt:
    def test_returns_grounded_prompt_instance(self):
        from src.evaluation.answer_correctness import (
            build_answer_correctness_judge_prompt,
        )

        prompt = build_answer_correctness_judge_prompt("Q?", "Expected.", "Actual.")

        assert isinstance(prompt, GroundedPrompt)

    def test_wraps_question_expected_actual_in_nonce_tags(self):
        from src.evaluation.answer_correctness import (
            build_answer_correctness_judge_prompt,
        )

        prompt = build_answer_correctness_judge_prompt(
            "Who founded Northwind?",
            "Jane Doe in 2015.",
            "Jane Doe founded it in 2015.",
        )

        question_match = re.search(
            r"<question-([0-9a-f]+)>.*?</question-\1>", prompt.user, re.DOTALL
        )
        expected_match = re.search(
            r"<expected-answer-([0-9a-f]+)>.*?</expected-answer-\1>",
            prompt.user,
            re.DOTALL,
        )
        actual_match = re.search(
            r"<actual-answer-([0-9a-f]+)>.*?</actual-answer-\1>", prompt.user, re.DOTALL
        )
        assert question_match is not None
        assert expected_match is not None
        assert actual_match is not None

    def test_nonce_differs_between_calls(self):
        from src.evaluation.answer_correctness import (
            build_answer_correctness_judge_prompt,
        )

        prompt1 = build_answer_correctness_judge_prompt("Q", "E", "A")
        prompt2 = build_answer_correctness_judge_prompt("Q", "E", "A")

        match1 = re.search(r"<question-([0-9a-f]+)>", prompt1.user)
        match2 = re.search(r"<question-([0-9a-f]+)>", prompt2.user)
        assert match1.group(1) != match2.group(1)


class TestScoreAnswerCorrectness:
    def test_delegates_to_judge_and_returns_verdict(self):
        from src.evaluation.answer_correctness import (
            CorrectnessVerdict,
            score_answer_correctness,
        )

        judge = FakeJudge(
            verdict=CorrectnessVerdict(correct=True, reasoning="Matches.")
        )

        verdict = score_answer_correctness("Q?", "Expected.", "Actual.", judge)

        assert verdict.correct is True
        assert verdict.reasoning == "Matches."

    def test_calls_judge_with_question_expected_actual(self):
        from src.evaluation.answer_correctness import score_answer_correctness

        judge = FakeJudge()

        score_answer_correctness(
            "Who founded it?", "Jane Doe.", "Jane Doe founded it.", judge
        )

        assert judge.calls == [("Who founded it?", "Jane Doe.", "Jane Doe founded it.")]

    def test_incorrect_verdict_propagates(self):
        from src.evaluation.answer_correctness import (
            CorrectnessVerdict,
            score_answer_correctness,
        )

        judge = FakeJudge(
            verdict=CorrectnessVerdict(correct=False, reasoning="Does not match.")
        )

        verdict = score_answer_correctness("Q?", "Expected.", "Wrong answer.", judge)

        assert verdict.correct is False


@pytest.fixture
def anthropic_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANSWER_CORRECTNESS_JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


@pytest.fixture
def openai_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANSWER_CORRECTNESS_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("ANSWER_CORRECTNESS_JUDGE_MODEL", "gpt-4o-2024-08-06")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestMakeAnswerCorrectnessJudge:
    def test_anthropic_provider_returns_anthropic_judge(self, anthropic_settings):
        from src.evaluation.answer_correctness import make_answer_correctness_judge
        from src.evaluation.providers.answer_correctness_judge_anthropic import (
            AnthropicAnswerCorrectnessJudge,
        )

        with patch("anthropic.Anthropic", return_value=MagicMock()):
            result = make_answer_correctness_judge(anthropic_settings)

        assert isinstance(result, AnthropicAnswerCorrectnessJudge)

    def test_anthropic_provider_id_reflects_resolved_model(self, anthropic_settings):
        from src.evaluation.answer_correctness import make_answer_correctness_judge

        assert anthropic_settings.answer_correctness_judge_model == "claude-sonnet-4-5"

        with patch("anthropic.Anthropic", return_value=MagicMock()):
            result = make_answer_correctness_judge(anthropic_settings)

        assert result.provider_id == "anthropic/claude-sonnet-4-5"

    def test_anthropic_substitutes_default_when_model_not_claude(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("ANSWER_CORRECTNESS_JUDGE_PROVIDER", "anthropic")
        monkeypatch.setenv("ANSWER_CORRECTNESS_JUDGE_MODEL", "gpt-4o-2024-08-06")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.evaluation.answer_correctness import make_answer_correctness_judge
        from src.evaluation.providers.answer_correctness_judge_anthropic import (
            DEFAULT_MODEL,
        )

        settings = Settings()

        with patch("anthropic.Anthropic", return_value=MagicMock()):
            result = make_answer_correctness_judge(settings)

        assert result.provider_id == f"anthropic/{DEFAULT_MODEL}"

    def test_openai_provider_returns_openai_judge(self, openai_settings):
        from src.evaluation.answer_correctness import make_answer_correctness_judge
        from src.evaluation.providers.answer_correctness_judge_openai import (
            OpenAIAnswerCorrectnessJudge,
        )

        with patch("openai.OpenAI", return_value=MagicMock()):
            result = make_answer_correctness_judge(openai_settings)

        assert isinstance(result, OpenAIAnswerCorrectnessJudge)

    def test_openai_substitutes_default_when_model_not_gpt(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("ANSWER_CORRECTNESS_JUDGE_PROVIDER", "openai")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.evaluation.answer_correctness import make_answer_correctness_judge
        from src.evaluation.providers.answer_correctness_judge_openai import (
            DEFAULT_MODEL,
        )

        settings = Settings()

        with patch("openai.OpenAI", return_value=MagicMock()):
            result = make_answer_correctness_judge(settings)

        assert result.provider_id == f"openai/{DEFAULT_MODEL}"

    def test_unknown_provider_raises_value_error(self, anthropic_settings):
        from src.evaluation.answer_correctness import make_answer_correctness_judge

        object.__setattr__(
            anthropic_settings, "answer_correctness_judge_provider", "bogus"
        )

        with pytest.raises(ValueError, match="bogus"):
            make_answer_correctness_judge(anthropic_settings)

    def test_provider_modules_not_imported_at_module_level(self):
        import sys

        sys.modules.pop(
            "src.evaluation.providers.answer_correctness_judge_anthropic", None
        )
        sys.modules.pop(
            "src.evaluation.providers.answer_correctness_judge_openai", None
        )
        sys.modules.pop("src.evaluation.answer_correctness", None)

        import src.evaluation.answer_correctness  # noqa: F401

        assert (
            "src.evaluation.providers.answer_correctness_judge_anthropic"
            not in sys.modules
        )
        assert (
            "src.evaluation.providers.answer_correctness_judge_openai"
            not in sys.modules
        )
