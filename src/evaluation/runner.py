"""Per-test-case eval orchestration — wires retrieval, generation, and all four metrics.

No `collect_spans()`/`persist_trace` wrapping: eval runs stay out of the
production trace store — 51 batch/offline traces per run would conflate with
live request traces in the trace-view UI (see the plan's design decision on
this). `span()` calls inside judge providers are harmless no-ops without an
active sink.

Citation accuracy reuses `verify_citations` directly rather than
reimplementing it — it's already the "do citations support claims?" metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.evaluation.answer_correctness import (
    AnswerCorrectnessJudgeProtocol,
    score_answer_correctness,
)
from src.evaluation.faithfulness import FaithfulnessJudgeProtocol, score_faithfulness
from src.evaluation.models import MetricAggregate, TestCaseResult
from src.evaluation.retrieval_relevance import score_retrieval_relevance
from src.generation.citation_verifier import CitationJudgeProtocol, verify_citations
from src.generation.prompts import build_grounded_prompt

if TYPE_CHECKING:
    from src.evaluation.dataset import GoldenCase
    from src.generation.answer_generator import AnswerGeneratorProtocol
    from src.retrieval.hybrid_retriever import HybridRetriever


def run_test_case(
    case: GoldenCase,
    retriever: HybridRetriever,
    generator: AnswerGeneratorProtocol,
    citation_judge: CitationJudgeProtocol,
    faithfulness_judge: FaithfulnessJudgeProtocol,
    correctness_judge: AnswerCorrectnessJudgeProtocol,
) -> TestCaseResult:
    """Run one golden case through the pipeline and score all four metrics.

    Any exception is caught and recorded on `TestCaseResult.error` instead of
    propagating, so one API hiccup doesn't abort a whole eval run.
    """
    try:
        hits = retriever.retrieve(case.question)
        answer = generator.generate(build_grounded_prompt(case.question, hits))

        citation_results = verify_citations(answer, hits, citation_judge)
        citation_accuracy = (
            sum(1 for r in citation_results if r.supported) / len(citation_results)
            if citation_results
            else None
        )

        faithfulness = score_faithfulness(answer, hits, faithfulness_judge)
        relevance = score_retrieval_relevance(
            case.source_documents, case.source_sections, hits
        )
        correctness = score_answer_correctness(
            case.question, case.expected_answer, answer, correctness_judge
        )

        return TestCaseResult(
            case_id=case.id,
            category=case.category,
            question=case.question,
            expected_answer=case.expected_answer,
            generated_answer=answer,
            answer_correct=correctness.correct,
            faithfulness_score=faithfulness.score,
            retrieval_relevance_score=relevance.score,
            citation_accuracy_score=citation_accuracy,
        )
    except Exception as exc:
        return TestCaseResult(
            case_id=case.id,
            category=case.category,
            question=case.question,
            expected_answer=case.expected_answer,
            error=str(exc),
        )


def run_eval(
    cases: list[GoldenCase],
    retriever: HybridRetriever,
    generator: AnswerGeneratorProtocol,
    citation_judge: CitationJudgeProtocol,
    faithfulness_judge: FaithfulnessJudgeProtocol,
    correctness_judge: AnswerCorrectnessJudgeProtocol,
) -> list[TestCaseResult]:
    """Run every case in *cases* through `run_test_case`, in order."""
    return [
        run_test_case(
            case,
            retriever,
            generator,
            citation_judge,
            faithfulness_judge,
            correctness_judge,
        )
        for case in cases
    ]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate_subset(results: list[TestCaseResult]) -> MetricAggregate:
    correctness_values = [
        1.0 if r.answer_correct else 0.0
        for r in results
        if r.answer_correct is not None
    ]
    faithfulness_values = [
        r.faithfulness_score for r in results if r.faithfulness_score is not None
    ]
    relevance_values = [
        r.retrieval_relevance_score
        for r in results
        if r.retrieval_relevance_score is not None
    ]
    citation_values = [
        r.citation_accuracy_score
        for r in results
        if r.citation_accuracy_score is not None
    ]
    return MetricAggregate(
        answer_correctness=_mean(correctness_values),
        faithfulness=_mean(faithfulness_values),
        retrieval_relevance=_mean(relevance_values),
        citation_accuracy=_mean(citation_values),
    )


def aggregate(
    results: list[TestCaseResult],
) -> tuple[MetricAggregate, dict[str, MetricAggregate]]:
    """Overall + per-category mean of each metric, skipping `None` scores."""
    overall = _aggregate_subset(results)
    categories = sorted({r.category for r in results})
    by_category = {
        category: _aggregate_subset([r for r in results if r.category == category])
        for category in categories
    }
    return overall, by_category
