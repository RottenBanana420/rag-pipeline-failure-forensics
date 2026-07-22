"""Unit tests for src/evaluation/models.py (TestCaseResult, MetricAggregate, EvalReport)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel


class TestTestCaseResult:
    def test_is_pydantic_model(self):
        from src.evaluation.models import TestCaseResult

        assert issubclass(TestCaseResult, BaseModel)

    def test_required_fields(self):
        from src.evaluation.models import TestCaseResult

        result = TestCaseResult(
            case_id="qa-001",
            category="lookup",
            question="Who founded Northwind?",
            expected_answer="Jane Doe founded Northwind in 2015.",
        )

        assert result.case_id == "qa-001"
        assert result.category == "lookup"
        assert result.question == "Who founded Northwind?"
        assert result.expected_answer == "Jane Doe founded Northwind in 2015."

    def test_optional_fields_default_to_none(self):
        from src.evaluation.models import TestCaseResult

        result = TestCaseResult(
            case_id="qa-001",
            category="lookup",
            question="Q",
            expected_answer="A",
        )

        assert result.generated_answer is None
        assert result.answer_correct is None
        assert result.faithfulness_score is None
        assert result.retrieval_relevance_score is None
        assert result.citation_accuracy_score is None
        assert result.error is None

    def test_can_set_all_score_fields(self):
        from src.evaluation.models import TestCaseResult

        result = TestCaseResult(
            case_id="qa-021",
            category="multi_hop",
            question="Q",
            expected_answer="A",
            generated_answer="Generated answer text [1].",
            answer_correct=True,
            faithfulness_score=0.8,
            retrieval_relevance_score=0.5,
            citation_accuracy_score=1.0,
        )

        assert result.generated_answer == "Generated answer text [1]."
        assert result.answer_correct is True
        assert result.faithfulness_score == 0.8
        assert result.retrieval_relevance_score == 0.5
        assert result.citation_accuracy_score == 1.0

    def test_can_set_error(self):
        from src.evaluation.models import TestCaseResult

        result = TestCaseResult(
            case_id="qa-001",
            category="lookup",
            question="Q",
            expected_answer="A",
            error="RateLimitError: too many requests",
        )

        assert result.error == "RateLimitError: too many requests"


class TestMetricAggregate:
    def test_is_pydantic_model(self):
        from src.evaluation.models import MetricAggregate

        assert issubclass(MetricAggregate, BaseModel)

    def test_all_fields_optional_and_default_none(self):
        from src.evaluation.models import MetricAggregate

        agg = MetricAggregate()

        assert agg.answer_correctness is None
        assert agg.faithfulness is None
        assert agg.retrieval_relevance is None
        assert agg.citation_accuracy is None

    def test_can_set_all_fields(self):
        from src.evaluation.models import MetricAggregate

        agg = MetricAggregate(
            answer_correctness=0.9,
            faithfulness=0.85,
            retrieval_relevance=0.7,
            citation_accuracy=0.95,
        )

        assert agg.answer_correctness == 0.9
        assert agg.faithfulness == 0.85
        assert agg.retrieval_relevance == 0.7
        assert agg.citation_accuracy == 0.95


class TestEvalReport:
    def test_is_pydantic_model(self):
        from src.evaluation.models import EvalReport

        assert issubclass(EvalReport, BaseModel)

    def test_construct_with_required_fields(self):
        from src.evaluation.models import EvalReport, MetricAggregate

        report = EvalReport(
            run_id="20260715T120000Z",
            timestamp=datetime.now(UTC),
            overall=MetricAggregate(),
        )

        assert report.run_id == "20260715T120000Z"
        assert report.results == []
        assert report.by_category == {}

    def test_construct_with_results_and_by_category(self):
        from src.evaluation.models import EvalReport, MetricAggregate, TestCaseResult

        result = TestCaseResult(
            case_id="qa-001", category="lookup", question="Q", expected_answer="A"
        )
        report = EvalReport(
            run_id="run-1",
            timestamp=datetime.now(UTC),
            results=[result],
            overall=MetricAggregate(answer_correctness=1.0),
            by_category={"lookup": MetricAggregate(answer_correctness=1.0)},
        )

        assert report.results == [result]
        assert report.by_category["lookup"].answer_correctness == 1.0

    def test_json_round_trip_preserves_nested_results(self):
        from src.evaluation.models import EvalReport, MetricAggregate, TestCaseResult

        result = TestCaseResult(
            case_id="qa-001",
            category="lookup",
            question="Q",
            expected_answer="A",
            generated_answer="Generated.",
            answer_correct=True,
            faithfulness_score=0.5,
            retrieval_relevance_score=None,
            citation_accuracy_score=1.0,
        )
        report = EvalReport(
            run_id="run-1",
            timestamp=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
            results=[result],
            overall=MetricAggregate(answer_correctness=1.0, citation_accuracy=1.0),
            by_category={"lookup": MetricAggregate(answer_correctness=1.0)},
        )

        loaded = EvalReport.model_validate_json(report.model_dump_json())

        assert loaded == report

    def test_json_round_trip_preserves_none_scores(self):
        from src.evaluation.models import EvalReport, MetricAggregate, TestCaseResult

        result = TestCaseResult(
            case_id="qa-032",
            category="no_answer",
            question="Q",
            expected_answer="I don't know.",
            retrieval_relevance_score=None,
        )
        report = EvalReport(
            run_id="run-2",
            timestamp=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
            results=[result],
            overall=MetricAggregate(),
        )

        loaded = EvalReport.model_validate_json(report.model_dump_json())

        assert loaded.results[0].retrieval_relevance_score is None
