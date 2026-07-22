"""Unit tests for src/evaluation/faithfulness.py — mirrors test_citation_verifier.py's shape."""

import dataclasses
import re
from unittest.mock import MagicMock, patch

import pytest

from src.generation.prompts import GroundedPrompt
from src.retrieval.models import VectorStoreHit


def make_hit(
    chunk_id: str = "chunk-1",
    text: str = "Paris is the capital of France.",
    doc_id: str = "doc-1",
    source_path: str = "/docs/geography.md",
    title: str = "Geography Facts",
    section_heading: str | None = "Capitals",
    chunk_index: int = 0,
    strategy: str = "fixed_size",
    similarity: float = 0.9,
) -> VectorStoreHit:
    return VectorStoreHit(
        chunk_id=chunk_id,
        text=text,
        doc_id=doc_id,
        source_path=source_path,
        title=title,
        section_heading=section_heading,
        chunk_index=chunk_index,
        strategy=strategy,
        similarity=similarity,
    )


class FakeJudge:
    def __init__(self, verdicts=None, provider_id: str = "fake/v1") -> None:
        self._verdicts = verdicts or {}
        self.calls: list[tuple[str, str]] = []
        self._provider_id = provider_id

    def judge(self, claim: str, context: str):
        from src.evaluation.faithfulness import FaithfulnessVerdict

        self.calls.append((claim, context))
        if claim in self._verdicts:
            return self._verdicts[claim]
        return FaithfulnessVerdict(grounded=True, reasoning="Default canned verdict.")

    @property
    def provider_id(self) -> str:
        return self._provider_id


class TestFaithfulnessJudgeProtocol:
    def test_fake_judge_satisfies_protocol(self):
        from src.evaluation.faithfulness import FaithfulnessJudgeProtocol

        assert isinstance(FakeJudge(), FaithfulnessJudgeProtocol)


class TestFaithfulnessVerdict:
    def test_is_pydantic_model(self):
        from pydantic import BaseModel

        from src.evaluation.faithfulness import FaithfulnessVerdict

        assert issubclass(FaithfulnessVerdict, BaseModel)

    def test_has_grounded_and_reasoning_fields(self):
        from src.evaluation.faithfulness import FaithfulnessVerdict

        verdict = FaithfulnessVerdict(grounded=True, reasoning="Matches context.")

        assert verdict.grounded is True
        assert verdict.reasoning == "Matches context."


class TestSplitIntoClaims:
    def test_splits_multiple_sentences(self):
        from src.evaluation.faithfulness import split_into_claims

        claims = split_into_claims("Paris is the capital. Lyon is a city.")

        assert claims == ["Paris is the capital.", "Lyon is a city."]

    def test_single_sentence_returns_one_claim(self):
        from src.evaluation.faithfulness import split_into_claims

        assert split_into_claims("Only one sentence here.") == [
            "Only one sentence here."
        ]

    def test_empty_string_returns_empty_list(self):
        from src.evaluation.faithfulness import split_into_claims

        assert split_into_claims("") == []

    def test_whitespace_only_returns_empty_list(self):
        from src.evaluation.faithfulness import split_into_claims

        assert split_into_claims("   \n  ") == []

    def test_handles_question_and_exclamation_marks(self):
        from src.evaluation.faithfulness import split_into_claims

        claims = split_into_claims("Is this true? Yes it is! Confirmed.")

        assert claims == ["Is this true?", "Yes it is!", "Confirmed."]


class TestBuildFaithfulnessJudgePrompt:
    def test_returns_grounded_prompt_instance(self):
        from src.evaluation.faithfulness import build_faithfulness_judge_prompt

        prompt = build_faithfulness_judge_prompt(
            "Paris is the capital.", "context text"
        )

        assert isinstance(prompt, GroundedPrompt)

    def test_claim_and_context_wrapped_in_nonce_tags(self):
        from src.evaluation.faithfulness import build_faithfulness_judge_prompt

        prompt = build_faithfulness_judge_prompt("The sky is blue.", "Sky facts.")

        claim_match = re.search(
            r"<claim-([0-9a-f]+)>.*?</claim-\1>", prompt.user, re.DOTALL
        )
        context_match = re.search(
            r"<context-([0-9a-f]+)>.*?</context-\1>", prompt.user, re.DOTALL
        )
        assert claim_match is not None
        assert context_match is not None

    def test_nonce_differs_between_calls(self):
        from src.evaluation.faithfulness import build_faithfulness_judge_prompt

        prompt1 = build_faithfulness_judge_prompt("claim", "context")
        prompt2 = build_faithfulness_judge_prompt("claim", "context")

        match1 = re.search(r"<claim-([0-9a-f]+)>", prompt1.user)
        match2 = re.search(r"<claim-([0-9a-f]+)>", prompt2.user)
        assert match1.group(1) != match2.group(1)


class TestScoreFaithfulness:
    def test_empty_hits_returns_none_score_without_calling_judge(self):
        from src.evaluation.faithfulness import score_faithfulness

        judge = FakeJudge()

        result = score_faithfulness("Some claim.", [], judge)

        assert result.claim_results == []
        assert result.score is None
        assert len(judge.calls) == 0

    def test_empty_answer_text_returns_none_score_without_calling_judge(self):
        from src.evaluation.faithfulness import score_faithfulness

        judge = FakeJudge()
        hits = [make_hit()]

        result = score_faithfulness("", hits, judge)

        assert result.score is None
        assert len(judge.calls) == 0

    def test_all_claims_grounded_gives_score_one(self):
        from src.evaluation.faithfulness import score_faithfulness

        judge = FakeJudge()
        hits = [make_hit(text="Paris is the capital of France.")]

        result = score_faithfulness("Paris is the capital.", hits, judge)

        assert result.score == pytest.approx(1.0)
        assert len(result.claim_results) == 1
        assert result.claim_results[0].grounded is True

    def test_mixed_claims_gives_fractional_score(self):
        from src.evaluation.faithfulness import FaithfulnessVerdict, score_faithfulness

        judge = FakeJudge(
            verdicts={
                "The moon is made of cheese.": FaithfulnessVerdict(
                    grounded=False, reasoning="Not supported by context."
                )
            }
        )
        hits = [make_hit(text="Paris is the capital of France.")]

        answer = "Paris is the capital. The moon is made of cheese."
        result = score_faithfulness(answer, hits, judge)

        assert result.score == pytest.approx(0.5)
        assert len(judge.calls) == 2

    def test_judge_called_with_full_concatenated_context_not_just_cited_chunk(self):
        from src.evaluation.faithfulness import score_faithfulness

        judge = FakeJudge()
        hits = [
            make_hit(chunk_id="a", text="CHUNK_ONE"),
            make_hit(chunk_id="b", text="CHUNK_TWO"),
        ]

        score_faithfulness("A single claim.", hits, judge)

        _, context = judge.calls[0]
        assert "CHUNK_ONE" in context
        assert "CHUNK_TWO" in context

    def test_calls_judge_once_per_claim_no_batching(self):
        from src.evaluation.faithfulness import score_faithfulness

        judge = FakeJudge()
        hits = [make_hit()]

        score_faithfulness("First claim. Second claim. Third claim.", hits, judge)

        assert len(judge.calls) == 3

    def test_records_verification_span(self):
        from src.evaluation.faithfulness import score_faithfulness
        from src.tracing.context import collect_spans

        judge = FakeJudge()
        hits = [make_hit()]

        with collect_spans() as spans:
            score_faithfulness("A claim.", hits, judge)

        assert len(spans) == 1
        assert spans[0].step == "verification"
        assert spans[0].error is None

    def test_confidence_score_all_grounded_is_5(self):
        from src.evaluation.faithfulness import score_faithfulness
        from src.tracing.context import collect_spans

        judge = FakeJudge()
        hits = [make_hit()]

        with collect_spans() as spans:
            score_faithfulness("A claim.", hits, judge)

        assert spans[0].confidence_score == 5

    def test_confidence_score_none_when_no_hits(self):
        from src.evaluation.faithfulness import score_faithfulness
        from src.tracing.context import collect_spans

        judge = FakeJudge()

        with collect_spans() as spans:
            score_faithfulness("A claim.", [], judge)

        assert spans[0].confidence_score is None

    def test_noop_outside_collect_spans(self):
        from src.evaluation.faithfulness import score_faithfulness

        judge = FakeJudge()
        score_faithfulness("A claim.", [make_hit()], judge)

    def test_insufficient_context_response_short_circuits_without_calling_judge(self):
        from src.evaluation.faithfulness import score_faithfulness
        from src.generation.prompts import INSUFFICIENT_CONTEXT_RESPONSE

        judge = FakeJudge()
        hits = [make_hit()]

        result = score_faithfulness(INSUFFICIENT_CONTEXT_RESPONSE, hits, judge)

        assert result.score is None
        assert result.claim_results == []
        assert len(judge.calls) == 0


class TestFaithfulnessResultIsFrozenDataclass:
    def test_frozen(self):
        from src.evaluation.faithfulness import FaithfulnessResult

        result = FaithfulnessResult(claim_results=[], score=None)

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.score = 1.0  # type: ignore[misc]


@pytest.fixture
def anthropic_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("FAITHFULNESS_JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


@pytest.fixture
def openai_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("FAITHFULNESS_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("FAITHFULNESS_JUDGE_MODEL", "gpt-4o-2024-08-06")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestMakeFaithfulnessJudge:
    def test_anthropic_provider_returns_anthropic_judge(self, anthropic_settings):
        from src.evaluation.faithfulness import make_faithfulness_judge
        from src.evaluation.providers.faithfulness_judge_anthropic import (
            AnthropicFaithfulnessJudge,
        )

        with patch("anthropic.Anthropic", return_value=MagicMock()):
            result = make_faithfulness_judge(anthropic_settings)

        assert isinstance(result, AnthropicFaithfulnessJudge)

    def test_anthropic_provider_id_reflects_resolved_model(self, anthropic_settings):
        from src.evaluation.faithfulness import make_faithfulness_judge

        assert anthropic_settings.faithfulness_judge_model == "claude-sonnet-4-5"

        with patch("anthropic.Anthropic", return_value=MagicMock()):
            result = make_faithfulness_judge(anthropic_settings)

        assert result.provider_id == "anthropic/claude-sonnet-4-5"

    def test_anthropic_substitutes_default_when_model_not_claude(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("FAITHFULNESS_JUDGE_PROVIDER", "anthropic")
        monkeypatch.setenv("FAITHFULNESS_JUDGE_MODEL", "gpt-4o-2024-08-06")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.evaluation.faithfulness import make_faithfulness_judge
        from src.evaluation.providers.faithfulness_judge_anthropic import DEFAULT_MODEL

        settings = Settings()

        with patch("anthropic.Anthropic", return_value=MagicMock()):
            result = make_faithfulness_judge(settings)

        assert result.provider_id == f"anthropic/{DEFAULT_MODEL}"

    def test_openai_provider_returns_openai_judge(self, openai_settings):
        from src.evaluation.faithfulness import make_faithfulness_judge
        from src.evaluation.providers.faithfulness_judge_openai import (
            OpenAIFaithfulnessJudge,
        )

        with patch("openai.OpenAI", return_value=MagicMock()):
            result = make_faithfulness_judge(openai_settings)

        assert isinstance(result, OpenAIFaithfulnessJudge)

    def test_openai_substitutes_default_when_model_not_gpt(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("FAITHFULNESS_JUDGE_PROVIDER", "openai")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.evaluation.faithfulness import make_faithfulness_judge
        from src.evaluation.providers.faithfulness_judge_openai import DEFAULT_MODEL

        settings = Settings()

        with patch("openai.OpenAI", return_value=MagicMock()):
            result = make_faithfulness_judge(settings)

        assert result.provider_id == f"openai/{DEFAULT_MODEL}"

    def test_unknown_provider_raises_value_error(self, anthropic_settings):
        from src.evaluation.faithfulness import make_faithfulness_judge

        object.__setattr__(anthropic_settings, "faithfulness_judge_provider", "bogus")

        with pytest.raises(ValueError, match="bogus"):
            make_faithfulness_judge(anthropic_settings)

    def test_provider_modules_not_imported_at_module_level(self):
        import sys

        sys.modules.pop("src.evaluation.providers.faithfulness_judge_anthropic", None)
        sys.modules.pop("src.evaluation.providers.faithfulness_judge_openai", None)
        sys.modules.pop("src.evaluation.faithfulness", None)

        import src.evaluation.faithfulness  # noqa: F401

        assert (
            "src.evaluation.providers.faithfulness_judge_anthropic" not in sys.modules
        )
        assert "src.evaluation.providers.faithfulness_judge_openai" not in sys.modules
