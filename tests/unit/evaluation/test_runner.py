"""Unit tests for src/evaluation/runner.py — fakes for retriever/generator/judges,
no real API calls (mirrors tests/unit/frontend/test_query_service.py's convention)."""

from __future__ import annotations

from src.evaluation.answer_correctness import CorrectnessVerdict
from src.evaluation.faithfulness import FaithfulnessVerdict
from src.generation.citation_verifier import JudgeVerdict
from src.retrieval.models import VectorStoreHit


def make_hit(
    chunk_id: str = "chunk-1",
    text: str = "Jane Doe founded Northwind in 2015.",
    doc_id: str = "doc-1",
    source_path: str = "/repo/data/golden/corpus/01-onboarding-guide.md",
    title: str = "Onboarding Guide",
    section_heading: str | None = "Welcome & Team Structure",
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


def make_case(
    id: str = "qa-001",
    question: str = "Who founded Northwind?",
    expected_answer: str = "Jane Doe founded Northwind in 2015.",
    category: str = "lookup",
    source_documents: list[str] | None = None,
    source_sections: list[str | None] | None = None,
):
    from src.evaluation.dataset import GoldenCase

    return GoldenCase(
        id=id,
        question=question,
        expected_answer=expected_answer,
        category=category,
        source_documents=source_documents
        if source_documents is not None
        else ["01-onboarding-guide.md"],
        source_sections=source_sections
        if source_sections is not None
        else ["Welcome & Team Structure"],
        notes=None,
    )


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
    def __init__(self, answer: str = "Jane Doe founded Northwind in 2015 [1]."):
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


class FakeFaithfulnessJudge:
    def __init__(self, grounded: bool = True):
        self._grounded = grounded
        self.calls = 0

    def judge(self, claim: str, context: str) -> FaithfulnessVerdict:
        self.calls += 1
        return FaithfulnessVerdict(grounded=self._grounded, reasoning="fake")

    @property
    def provider_id(self) -> str:
        return "fake-faithfulness-judge/v1"


class FakeCorrectnessJudge:
    def __init__(self, correct: bool = True):
        self._correct = correct
        self.calls = 0

    def judge(
        self, question: str, expected_answer: str, actual_answer: str
    ) -> CorrectnessVerdict:
        self.calls += 1
        return CorrectnessVerdict(correct=self._correct, reasoning="fake")

    @property
    def provider_id(self) -> str:
        return "fake-correctness-judge/v1"


class TestRunTestCase:
    def test_success_path_populates_all_scores(self):
        from src.evaluation.runner import run_test_case

        case = make_case()
        result = run_test_case(
            case,
            retriever=FakeRetriever(),
            generator=FakeAnswerGenerator(),
            citation_judge=FakeCitationJudge(supported=True),
            faithfulness_judge=FakeFaithfulnessJudge(grounded=True),
            correctness_judge=FakeCorrectnessJudge(correct=True),
        )

        assert result.case_id == "qa-001"
        assert result.category == "lookup"
        assert result.generated_answer == "Jane Doe founded Northwind in 2015 [1]."
        assert result.answer_correct is True
        assert result.faithfulness_score == 1.0
        assert result.retrieval_relevance_score == 1.0
        assert result.citation_accuracy_score == 1.0
        assert result.error is None

    def test_no_citations_in_answer_gives_none_citation_accuracy(self):
        from src.evaluation.runner import run_test_case

        case = make_case()
        result = run_test_case(
            case,
            retriever=FakeRetriever(),
            generator=FakeAnswerGenerator(answer="No markers here."),
            citation_judge=FakeCitationJudge(),
            faithfulness_judge=FakeFaithfulnessJudge(),
            correctness_judge=FakeCorrectnessJudge(),
        )

        assert result.citation_accuracy_score is None

    def test_no_answer_category_gives_none_retrieval_relevance(self):
        from src.evaluation.runner import run_test_case

        case = make_case(
            id="qa-032",
            category="no_answer",
            source_documents=[],
            source_sections=[],
        )
        result = run_test_case(
            case,
            retriever=FakeRetriever(hits=[]),
            generator=FakeAnswerGenerator(answer="I don't have enough information."),
            citation_judge=FakeCitationJudge(),
            faithfulness_judge=FakeFaithfulnessJudge(),
            correctness_judge=FakeCorrectnessJudge(),
        )

        assert result.retrieval_relevance_score is None
        assert result.faithfulness_score is None

    def test_retriever_exception_is_caught_and_recorded_as_error(self):
        from src.evaluation.runner import run_test_case

        case = make_case()
        result = run_test_case(
            case,
            retriever=FakeRetriever(error=RuntimeError("connection refused")),
            generator=FakeAnswerGenerator(),
            citation_judge=FakeCitationJudge(),
            faithfulness_judge=FakeFaithfulnessJudge(),
            correctness_judge=FakeCorrectnessJudge(),
        )

        assert result.error == "connection refused"
        assert result.generated_answer is None
        assert result.answer_correct is None

    def test_retriever_called_with_question(self):
        from src.evaluation.runner import run_test_case

        case = make_case(question="Who founded Northwind?")
        retriever = FakeRetriever()
        run_test_case(
            case,
            retriever=retriever,
            generator=FakeAnswerGenerator(),
            citation_judge=FakeCitationJudge(),
            faithfulness_judge=FakeFaithfulnessJudge(),
            correctness_judge=FakeCorrectnessJudge(),
        )

        assert retriever.calls == ["Who founded Northwind?"]


class TestRunEval:
    def test_iterates_all_cases(self):
        from src.evaluation.runner import run_eval

        cases = [make_case(id="qa-001"), make_case(id="qa-002")]
        results = run_eval(
            cases,
            retriever=FakeRetriever(),
            generator=FakeAnswerGenerator(),
            citation_judge=FakeCitationJudge(),
            faithfulness_judge=FakeFaithfulnessJudge(),
            correctness_judge=FakeCorrectnessJudge(),
        )

        assert [r.case_id for r in results] == ["qa-001", "qa-002"]

    def test_one_case_erroring_does_not_abort_the_run(self):
        from src.evaluation.runner import run_eval

        class FlakyRetriever:
            def __init__(self):
                self.calls = 0

            def retrieve(self, query: str):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("boom")
                return [make_hit()]

        cases = [make_case(id="qa-001"), make_case(id="qa-002")]
        results = run_eval(
            cases,
            retriever=FlakyRetriever(),
            generator=FakeAnswerGenerator(),
            citation_judge=FakeCitationJudge(),
            faithfulness_judge=FakeFaithfulnessJudge(),
            correctness_judge=FakeCorrectnessJudge(),
        )

        assert results[0].error == "boom"
        assert results[1].error is None


class TestAggregate:
    def test_overall_means_skip_none_scores(self):
        from src.evaluation.models import TestCaseResult
        from src.evaluation.runner import aggregate

        results = [
            TestCaseResult(
                case_id="qa-001",
                category="lookup",
                question="Q1",
                expected_answer="A1",
                answer_correct=True,
                faithfulness_score=1.0,
                retrieval_relevance_score=1.0,
                citation_accuracy_score=1.0,
            ),
            TestCaseResult(
                case_id="qa-032",
                category="no_answer",
                question="Q2",
                expected_answer="A2",
                answer_correct=True,
                faithfulness_score=None,
                retrieval_relevance_score=None,
                citation_accuracy_score=None,
            ),
        ]

        overall, _ = aggregate(results)

        assert overall.answer_correctness == 1.0
        assert overall.faithfulness == 1.0
        assert overall.retrieval_relevance == 1.0
        assert overall.citation_accuracy == 1.0

    def test_per_category_breakdown(self):
        from src.evaluation.models import TestCaseResult
        from src.evaluation.runner import aggregate

        results = [
            TestCaseResult(
                case_id="qa-001",
                category="lookup",
                question="Q1",
                expected_answer="A1",
                answer_correct=True,
            ),
            TestCaseResult(
                case_id="qa-002",
                category="lookup",
                question="Q2",
                expected_answer="A2",
                answer_correct=False,
            ),
            TestCaseResult(
                case_id="qa-021",
                category="multi_hop",
                question="Q3",
                expected_answer="A3",
                answer_correct=True,
            ),
        ]

        _, by_category = aggregate(results)

        assert by_category["lookup"].answer_correctness == 0.5
        assert by_category["multi_hop"].answer_correctness == 1.0

    def test_empty_results_gives_none_aggregates(self):
        from src.evaluation.runner import aggregate

        overall, by_category = aggregate([])

        assert overall.answer_correctness is None
        assert overall.faithfulness is None
        assert overall.retrieval_relevance is None
        assert overall.citation_accuracy is None
        assert by_category == {}

    def test_error_only_case_excluded_from_all_aggregates(self):
        from src.evaluation.models import TestCaseResult
        from src.evaluation.runner import aggregate

        results = [
            TestCaseResult(
                case_id="qa-001",
                category="lookup",
                question="Q1",
                expected_answer="A1",
                error="boom",
            )
        ]

        overall, by_category = aggregate(results)

        assert overall.answer_correctness is None
        assert by_category["lookup"].answer_correctness is None
