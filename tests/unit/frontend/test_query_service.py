"""Unit tests for the ask_question orchestrator, with fakes standing in for the
retriever/generator/judges (mirrors tests/unit/frontend/test_diagnosis_service.py's
fake-judge pattern) — no real API calls."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.config import Settings
from src.generation.citation_verifier import JudgeVerdict
from src.generation.confidence_scorer import CompletenessVerdict
from src.retrieval.models import VectorStoreHit
from src.tracing.index import get_trace_record, init_trace_index


def make_hit(
    chunk_id: str = "chunk-1",
    text: str = "The on-call rotation is weekly.",
    doc_id: str = "doc-1",
    source_path: str = "/docs/oncall.md",
    title: str = "On-call Runbook",
    section_heading: str | None = "Rotation",
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


def settings_for(tmp_path, **overrides) -> Settings:
    base = dict(
        trace_output_dir=tmp_path / "traces",
        sqlite_db_path=tmp_path / "traces.db",
        retrieval_confidence_threshold=0.5,
    )
    base.update(overrides)
    return Settings(**base)


class FakeRetriever:
    def __init__(
        self, hits: list[VectorStoreHit] | None = None, error: Exception | None = None
    ):
        self._hits = hits if hits is not None else [make_hit()]
        self._error = error
        self.calls: list[str] = []

    def retrieve(self, query: str) -> list[VectorStoreHit]:
        self.calls.append(query)
        if self._error is not None:
            raise self._error
        return self._hits


class FakeAnswerGenerator:
    def __init__(self, answer: str = "The on-call rotation is weekly [1]."):
        self._answer = answer
        self.calls = 0

    def generate(self, prompt) -> str:
        self.calls += 1
        return self._answer

    @property
    def provider_id(self) -> str:
        return "fake-generator/v1"


class FakeCitationJudge:
    def __init__(self, supported: bool = True):
        self._supported = supported
        self.calls = 0

    def judge(self, claim: str, evidence: str) -> JudgeVerdict:
        self.calls += 1
        return JudgeVerdict(supported=self._supported, reasoning="fake")

    @property
    def provider_id(self) -> str:
        return "fake-citation-judge/v1"


class FakeCompletenessJudge:
    def __init__(self, complete: bool = True):
        self._complete = complete
        self.calls = 0

    def judge(self, question: str, answer: str) -> CompletenessVerdict:
        self.calls += 1
        return CompletenessVerdict(complete=self._complete, reasoning="fake")

    @property
    def provider_id(self) -> str:
        return "fake-completeness-judge/v1"


def _patched(generator, citation_judge, completeness_judge):
    return (
        patch(
            "src.frontend.query_service.make_answer_generator",
            return_value=generator,
        ),
        patch(
            "src.frontend.query_service.make_citation_judge",
            return_value=citation_judge,
        ),
        patch(
            "src.frontend.query_service.make_completeness_judge",
            return_value=completeness_judge,
        ),
    )


class TestAskQuestionSuccessPath:
    def test_returns_result_with_answer_and_success_status(self, tmp_path):
        from src.frontend.query_service import ask_question

        settings = settings_for(tmp_path)
        retriever = FakeRetriever(hits=[make_hit(similarity=0.9)])
        generator = FakeAnswerGenerator("The on-call rotation is weekly [1].")
        citation_judge = FakeCitationJudge(supported=True)
        completeness_judge = FakeCompletenessJudge(complete=True)

        p1, p2, p3 = _patched(generator, citation_judge, completeness_judge)
        with p1, p2, p3:
            result = ask_question("How often does on-call rotate?", retriever, settings)

        assert result.status == "success"
        assert result.answer_text == "The on-call rotation is weekly [1]."
        assert result.fallback is None
        assert result.hits == [make_hit(similarity=0.9)]
        assert result.confidence.retrieval_confidence == pytest.approx(0.9)
        assert len(result.citation_results) == 1
        assert result.citation_results[0].supported is True

    def test_persists_success_trace_matching_returned_trace_id(self, tmp_path):
        from src.frontend.query_service import ask_question

        settings = settings_for(tmp_path)
        retriever = FakeRetriever()
        generator = FakeAnswerGenerator()
        citation_judge = FakeCitationJudge()
        completeness_judge = FakeCompletenessJudge()

        p1, p2, p3 = _patched(generator, citation_judge, completeness_judge)
        with p1, p2, p3:
            result = ask_question("question", retriever, settings)

        init_trace_index(settings.sqlite_db_path)
        record = get_trace_record(result.trace_id, settings.sqlite_db_path)
        assert record is not None
        assert record.status == "success"

    def test_calls_generator_and_each_judge_exactly_once(self, tmp_path):
        from src.frontend.query_service import ask_question

        settings = settings_for(tmp_path)
        retriever = FakeRetriever()
        generator = FakeAnswerGenerator("Weekly [1].")
        citation_judge = FakeCitationJudge()
        completeness_judge = FakeCompletenessJudge()

        p1, p2, p3 = _patched(generator, citation_judge, completeness_judge)
        with p1, p2, p3:
            ask_question("question", retriever, settings)

        assert generator.calls == 1
        assert citation_judge.calls == 1
        assert completeness_judge.calls == 1


class TestAskQuestionFallbackPath:
    def test_low_retrieval_confidence_triggers_fallback_and_degraded_status(
        self, tmp_path
    ):
        from src.frontend.query_service import ask_question

        settings = settings_for(tmp_path, retrieval_confidence_threshold=0.5)
        retriever = FakeRetriever(hits=[make_hit(similarity=0.1)])
        generator = FakeAnswerGenerator("Weekly [1].")
        citation_judge = FakeCitationJudge()
        completeness_judge = FakeCompletenessJudge()

        p1, p2, p3 = _patched(generator, citation_judge, completeness_judge)
        with p1, p2, p3:
            result = ask_question("question", retriever, settings)

        assert result.status == "degraded"
        assert result.fallback is not None
        # The raw generated text is still surfaced for engineers, even though
        # the dashboard itself will prefer the fallback message.
        assert result.answer_text == "Weekly [1]."

    def test_persists_degraded_trace(self, tmp_path):
        from src.frontend.query_service import ask_question

        settings = settings_for(tmp_path, retrieval_confidence_threshold=0.5)
        retriever = FakeRetriever(hits=[make_hit(similarity=0.1)])
        generator = FakeAnswerGenerator()
        citation_judge = FakeCitationJudge()
        completeness_judge = FakeCompletenessJudge()

        p1, p2, p3 = _patched(generator, citation_judge, completeness_judge)
        with p1, p2, p3:
            result = ask_question("question", retriever, settings)

        init_trace_index(settings.sqlite_db_path)
        record = get_trace_record(result.trace_id, settings.sqlite_db_path)
        assert record is not None
        assert record.status == "degraded"


class TestAskQuestionFailurePath:
    def test_exception_during_retrieval_persists_failure_trace_and_reraises(
        self, tmp_path
    ):
        from src.frontend.query_service import ask_question

        settings = settings_for(tmp_path)
        retriever = FakeRetriever(error=RuntimeError("vector store unavailable"))
        generator = FakeAnswerGenerator()
        citation_judge = FakeCitationJudge()
        completeness_judge = FakeCompletenessJudge()

        p1, p2, p3 = _patched(generator, citation_judge, completeness_judge)
        with p1, p2, p3, pytest.raises(RuntimeError, match="vector store unavailable"):
            ask_question("question", retriever, settings)

        init_trace_index(settings.sqlite_db_path)
        from src.tracing.index import list_trace_records

        records = list_trace_records(settings.sqlite_db_path, status="failure")
        assert len(records) == 1

    def test_generator_never_called_when_retrieval_raises(self, tmp_path):
        from src.frontend.query_service import ask_question

        settings = settings_for(tmp_path)
        retriever = FakeRetriever(error=RuntimeError("boom"))
        generator = FakeAnswerGenerator()
        citation_judge = FakeCitationJudge()
        completeness_judge = FakeCompletenessJudge()

        p1, p2, p3 = _patched(generator, citation_judge, completeness_judge)
        with p1, p2, p3, pytest.raises(RuntimeError):
            ask_question("question", retriever, settings)

        assert generator.calls == 0


class TestBuildHybridRetriever:
    """Every heavy dependency (embedder model, vector store, BM25 pickle,
    reranker) is patched at the module level — build_hybrid_retriever's job
    is wiring, not the dependencies' own construction, which each already
    have their own tests (test_embedder_factory.py, test_vector_store.py,
    etc.)."""

    def test_wires_dense_and_hybrid_retriever(self, tmp_path):
        from src.frontend.query_service import build_hybrid_retriever
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever

        settings = Settings(
            chroma_persist_dir=tmp_path / "chroma", reranking_enabled=False
        )
        fake_embedder = object()
        fake_vector_store = object()

        with (
            patch(
                "src.frontend.query_service.make_embedder",
                return_value=fake_embedder,
            ),
            patch(
                "src.frontend.query_service.make_vector_store",
                return_value=fake_vector_store,
            ),
            patch("src.frontend.query_service.BM25Store.load"),
        ):
            bundle = build_hybrid_retriever(settings)

        assert isinstance(bundle.hybrid, HybridRetriever)
        assert isinstance(bundle.dense, DenseRetriever)

    def test_no_reranker_when_reranking_disabled(self, tmp_path):
        from src.frontend.query_service import build_hybrid_retriever

        settings = Settings(
            chroma_persist_dir=tmp_path / "chroma", reranking_enabled=False
        )

        with (
            patch("src.frontend.query_service.make_embedder", return_value=object()),
            patch(
                "src.frontend.query_service.make_vector_store",
                return_value=object(),
            ),
            patch("src.frontend.query_service.BM25Store.load"),
            patch("src.frontend.query_service.make_reranker") as mock_make_reranker,
        ):
            bundle = build_hybrid_retriever(settings)

        mock_make_reranker.assert_not_called()
        assert bundle.hybrid._reranker is None

    def test_reranker_built_when_reranking_enabled(self, tmp_path):
        from src.frontend.query_service import build_hybrid_retriever

        settings = Settings(
            chroma_persist_dir=tmp_path / "chroma", reranking_enabled=True
        )
        fake_reranker = object()

        with (
            patch("src.frontend.query_service.make_embedder", return_value=object()),
            patch(
                "src.frontend.query_service.make_vector_store",
                return_value=object(),
            ),
            patch("src.frontend.query_service.BM25Store.load"),
            patch(
                "src.frontend.query_service.make_reranker",
                return_value=fake_reranker,
            ) as mock_make_reranker,
        ):
            bundle = build_hybrid_retriever(settings)

        mock_make_reranker.assert_called_once_with(settings)
        assert bundle.hybrid._reranker is fake_reranker
