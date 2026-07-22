"""Result models for the golden-dataset eval harness (Phase 6).

`TestCaseResult`/`EvalReport` are pydantic `BaseModel`s (not dataclasses) so
`EvalReport` can round-trip through `model_dump_json()`/`model_validate_json()`
for regression-tracking persistence — the same rationale `Trace`/`Span`
(`src/tracing/models.py`) use for the same JSON-round-trip need.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TestCaseResult(BaseModel):
    case_id: str
    category: str
    question: str
    expected_answer: str
    generated_answer: str | None = None
    answer_correct: bool | None = None
    faithfulness_score: float | None = None
    retrieval_relevance_score: float | None = None
    citation_accuracy_score: float | None = None
    error: str | None = None


class MetricAggregate(BaseModel):
    answer_correctness: float | None = None
    faithfulness: float | None = None
    retrieval_relevance: float | None = None
    citation_accuracy: float | None = None


class EvalReport(BaseModel):
    run_id: str
    timestamp: datetime
    results: list[TestCaseResult] = Field(default_factory=list)
    overall: MetricAggregate
    by_category: dict[str, MetricAggregate] = Field(default_factory=dict)
