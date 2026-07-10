"""Unit tests for the on-demand root-cause diagnosis service, with fake
judges standing in for real LLM calls (mirrors
tests/unit/analysis/test_root_cause.py's FakeStepQualityJudge pattern)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.analysis.evidence_chain import EvidenceChainVerdict
from src.analysis.failure_categorizer import FailureCategoryVerdict
from src.analysis.root_cause import StepQualityVerdict
from src.frontend.diagnosis_service import DiagnosisResult, run_diagnosis
from src.tracing.models import PipelineStep, Span, Trace


def fake_settings(root_cause_quality_threshold: int = 2) -> SimpleNamespace:
    """A minimal settings stand-in — run_diagnosis only reads
    `root_cause_quality_threshold` directly (the rest flows opaquely through
    to the patched judge factories), so a real Settings() isn't needed here."""
    return SimpleNamespace(root_cause_quality_threshold=root_cause_quality_threshold)


def make_span(
    step: PipelineStep = "retrieval",
    input: str = "in",
    output: str = "out",
    **overrides: object,
) -> Span:
    base: dict[str, object] = {
        "step": step,
        "input": input,
        "output": output,
        "latency_ms": 1.0,
    }
    base.update(overrides)
    return Span(**base)


class FakeStepQualityJudge:
    def __init__(self, verdicts: list[StepQualityVerdict]) -> None:
        self._verdicts = list(verdicts)
        self._index = 0
        self.calls: list[tuple[PipelineStep, str, str]] = []

    def judge(self, step: PipelineStep, input: str, output: str) -> StepQualityVerdict:
        self.calls.append((step, input, output))
        verdict = self._verdicts[self._index]
        self._index += 1
        return verdict

    @property
    def provider_id(self) -> str:
        return "fake-step-quality/v1"


class FakeFailureCategoryJudge:
    def __init__(self, verdict: FailureCategoryVerdict) -> None:
        self._verdict = verdict
        self.calls: list[tuple] = []

    def classify(
        self, step, input, output, quality_rationale
    ) -> FailureCategoryVerdict:
        self.calls.append((step, input, output, quality_rationale))
        return self._verdict

    @property
    def provider_id(self) -> str:
        return "fake-failure-category/v1"


class FakeEvidenceChainJudge:
    def __init__(self, verdict: EvidenceChainVerdict) -> None:
        self._verdict = verdict
        self.calls: list[tuple] = []

    def narrate(self, category, category_rationale, chain) -> EvidenceChainVerdict:
        self.calls.append((category, category_rationale, chain))
        return self._verdict

    @property
    def provider_id(self) -> str:
        return "fake-evidence-chain/v1"


class TestRunDiagnosis:
    def test_no_root_cause_found_short_circuits_before_category_and_narrative(self):
        """The cost-control-relevant test: when find_root_cause_span finds
        nothing, categorize_failure/build_evidence_chain must never run —
        that would be two wasted LLM calls for a healthy trace."""
        span = make_span(step="generation")
        trace = Trace(spans=[span], status="success")
        step_quality_judge = FakeStepQualityJudge(
            verdicts=[StepQualityVerdict(score=5, rationale="Fine.")]
        )
        category_judge = FakeFailureCategoryJudge(
            verdict=FailureCategoryVerdict(category="other", rationale="unused")
        )
        evidence_chain_judge = FakeEvidenceChainJudge(
            verdict=EvidenceChainVerdict(narrative="unused")
        )

        with (
            patch(
                "src.frontend.diagnosis_service.make_step_quality_judge",
                return_value=step_quality_judge,
            ),
            patch(
                "src.frontend.diagnosis_service.make_failure_category_judge",
                return_value=category_judge,
            ) as mock_make_category,
            patch(
                "src.frontend.diagnosis_service.make_evidence_chain_judge",
                return_value=evidence_chain_judge,
            ) as mock_make_evidence,
        ):
            result = run_diagnosis(trace, settings=fake_settings())  # type: ignore[arg-type]

        assert result == DiagnosisResult(
            diagnosis=None, category=None, evidence_chain=None
        )
        mock_make_category.assert_not_called()
        mock_make_evidence.assert_not_called()
        assert category_judge.calls == []
        assert evidence_chain_judge.calls == []

    def test_root_cause_found_invokes_all_three_judges_once_each(self):
        bad_span = make_span(step="retrieval", input="bad-in", output="bad-out")
        trace = Trace(spans=[bad_span], status="failure")
        step_quality_judge = FakeStepQualityJudge(
            verdicts=[StepQualityVerdict(score=1, rationale="Unreasonable.")]
        )
        category_verdict = FailureCategoryVerdict(
            category="retrieval_failure", rationale="No relevant chunks."
        )
        category_judge = FakeFailureCategoryJudge(verdict=category_verdict)
        evidence_verdict = EvidenceChainVerdict(narrative="Retrieval failed first.")
        evidence_chain_judge = FakeEvidenceChainJudge(verdict=evidence_verdict)

        with (
            patch(
                "src.frontend.diagnosis_service.make_step_quality_judge",
                return_value=step_quality_judge,
            ),
            patch(
                "src.frontend.diagnosis_service.make_failure_category_judge",
                return_value=category_judge,
            ),
            patch(
                "src.frontend.diagnosis_service.make_evidence_chain_judge",
                return_value=evidence_chain_judge,
            ),
        ):
            result = run_diagnosis(trace, settings=fake_settings())  # type: ignore[arg-type]

        assert result.diagnosis is not None
        assert result.diagnosis.root_cause_span is bad_span
        assert result.category == category_verdict
        assert result.evidence_chain is not None
        assert result.evidence_chain.narrative == "Retrieval failed first."
        assert len(category_judge.calls) == 1
        assert len(evidence_chain_judge.calls) == 1

    def test_judges_constructed_from_passed_settings(self):
        span = make_span(step="generation")
        trace = Trace(spans=[span], status="success")
        step_quality_judge = FakeStepQualityJudge(
            verdicts=[StepQualityVerdict(score=5, rationale="Fine.")]
        )
        sentinel_settings = fake_settings()

        with patch(
            "src.frontend.diagnosis_service.make_step_quality_judge",
            return_value=step_quality_judge,
        ) as mock_make_step_quality:
            run_diagnosis(trace, settings=sentinel_settings)  # type: ignore[arg-type]

        mock_make_step_quality.assert_called_once_with(sentinel_settings)
