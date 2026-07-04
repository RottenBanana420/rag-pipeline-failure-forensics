# Trace/Span Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Span` and `Trace` pydantic models to `src/tracing/models.py` — the data model that becomes the complete record of what happened during a RAG request.

**Architecture:** Two plain (non-frozen) pydantic `BaseModel`s in one new file, `src/tracing/models.py`, following the exact convention already used by `src/ingestion/models.py` (`ProcessedDocument`, `Chunk`): `Literal` type aliases for closed enumerations, `Field()` for numeric constraints, `default_factory` for auto-generated IDs. No context manager, decorator, or JSON/SQLite writer — those are separate future tasks that will consume these models.

**Tech Stack:** Python 3.11+, pydantic >=2.0, pytest.

## Global Constraints

- Python >=3.11, pydantic >=2.0 (from `pyproject.toml`).
- Follow existing model conventions in `src/ingestion/models.py` — pydantic `BaseModel`, `from __future__ import annotations`, `Literal` type aliases for closed sets, `Field(ge=..., le=...)` for numeric constraints.
- Follow existing test conventions in `tests/unit/ingestion/test_models.py` — a `_valid_kwargs()` helper returning a dict of valid constructor kwargs, `pytest.raises(ValidationError)` for rejection cases, a `test_round_trip_json` test using `model_dump_json()`/`model_validate_json()`.
- No context manager, decorator, JSON writer, or SQLite writer in this plan — models only.
- `input`/`output` on `Span` are plain `str` (already-serialized values) — do not attempt to type them as `Any` or a union of pipeline types.

---

### Task 1: `Span` model

**Files:**
- Create: `src/tracing/models.py`
- Create: `tests/unit/tracing/__init__.py` (empty file, matches `tests/unit/ingestion/__init__.py`)
- Test: `tests/unit/tracing/test_models.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `PipelineStep` (`Literal["ingestion", "retrieval", "ranking", "generation", "verification"]`), `Span` pydantic model with fields `span_id: str`, `step: PipelineStep`, `input: str`, `output: str`, `llm_prompt: str | None`, `token_count: int | None`, `latency_ms: float`, `confidence_score: int | None`. Task 2 imports `Span` from `src.tracing.models` to type `Trace.spans`.

- [ ] **Step 1: Create the empty test package `__init__.py`**

```bash
mkdir -p "tests/unit/tracing" && touch "tests/unit/tracing/__init__.py"
```

- [ ] **Step 2: Write the failing tests for `Span`**

Create `tests/unit/tracing/test_models.py` with this content:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.tracing.models import Span


def _valid_span_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "step": "retrieval",
        "input": '{"question": "What is RRF?"}',
        "output": '{"hits": []}',
        "latency_ms": 12.5,
    }
    base.update(overrides)
    return base


class TestSpanValidation:
    def test_valid_minimal_span(self):
        span = Span(**_valid_span_kwargs())
        assert span.step == "retrieval"
        assert span.llm_prompt is None
        assert span.token_count is None
        assert span.confidence_score is None
        assert isinstance(span.span_id, str) and span.span_id

    def test_span_id_auto_generated_and_unique(self):
        span_a = Span(**_valid_span_kwargs())
        span_b = Span(**_valid_span_kwargs())
        assert span_a.span_id != span_b.span_id

    def test_all_pipeline_steps_accepted(self):
        for step in (
            "ingestion",
            "retrieval",
            "ranking",
            "generation",
            "verification",
        ):
            span = Span(**_valid_span_kwargs(step=step))
            assert span.step == step

    def test_invalid_step_rejected(self):
        with pytest.raises(ValidationError):
            Span(**_valid_span_kwargs(step="not_a_step"))

    def test_optional_fields_populated(self):
        span = Span(
            **_valid_span_kwargs(
                step="generation",
                llm_prompt="Answer using only the provided context.",
                token_count=250,
                confidence_score=4,
            )
        )
        assert span.llm_prompt == "Answer using only the provided context."
        assert span.token_count == 250
        assert span.confidence_score == 4

    def test_confidence_score_below_range_rejected(self):
        with pytest.raises(ValidationError):
            Span(**_valid_span_kwargs(confidence_score=0))

    def test_confidence_score_above_range_rejected(self):
        with pytest.raises(ValidationError):
            Span(**_valid_span_kwargs(confidence_score=6))

    def test_negative_token_count_rejected(self):
        with pytest.raises(ValidationError):
            Span(**_valid_span_kwargs(token_count=-1))

    def test_negative_latency_rejected(self):
        with pytest.raises(ValidationError):
            Span(**_valid_span_kwargs(latency_ms=-0.1))

    def test_round_trip_json(self):
        span = Span(**_valid_span_kwargs(confidence_score=5, token_count=100))
        restored = Span.model_validate_json(span.model_dump_json())
        assert restored == span
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/unit/tracing/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tracing.models'` (or collection error) — the module doesn't exist yet.

- [ ] **Step 4: Implement `Span` in `src/tracing/models.py`**

```python
from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

PipelineStep = Literal["ingestion", "retrieval", "ranking", "generation", "verification"]


class Span(BaseModel):
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    step: PipelineStep
    input: str
    output: str
    llm_prompt: str | None = None
    token_count: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0.0)
    confidence_score: int | None = Field(default=None, ge=1, le=5)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/unit/tracing/test_models.py -v`
Expected: PASS (all `TestSpanValidation` tests green)

- [ ] **Step 6: Commit**

```bash
git add src/tracing/models.py tests/unit/tracing/__init__.py tests/unit/tracing/test_models.py
git commit -m "feat(tracing): add Span pydantic model"
```

---

### Task 2: `Trace` model

**Files:**
- Modify: `src/tracing/models.py` (append `TraceStatus` and `Trace`)
- Test: `tests/unit/tracing/test_models.py` (append `TestTraceValidation`)

**Interfaces:**
- Consumes: `Span` (from Task 1, `src.tracing.models.Span`), `PipelineStep` (unused directly but shares the module).
- Produces: `TraceStatus` (`Literal["success", "failure", "degraded"]`), `Trace` pydantic model with fields `trace_id: str`, `spans: list[Span]`, `final_output: str | None`, `status: TraceStatus`. No later task in this plan consumes `Trace`, but future context-manager/writer tasks (out of scope here) will.

- [ ] **Step 1: Write the failing tests for `Trace`**

Append this to `tests/unit/tracing/test_models.py` (add `Trace` to the existing `from src.tracing.models import Span` import line, making it `from src.tracing.models import Span, Trace`):

```python
def _valid_trace_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {"status": "success"}
    base.update(overrides)
    return base


class TestTraceValidation:
    def test_valid_minimal_trace(self):
        trace = Trace(**_valid_trace_kwargs())
        assert trace.status == "success"
        assert trace.spans == []
        assert trace.final_output is None
        assert isinstance(trace.trace_id, str) and trace.trace_id

    def test_trace_id_auto_generated_and_unique(self):
        trace_a = Trace(**_valid_trace_kwargs())
        trace_b = Trace(**_valid_trace_kwargs())
        assert trace_a.trace_id != trace_b.trace_id

    def test_all_statuses_accepted(self):
        for status in ("success", "failure", "degraded"):
            trace = Trace(**_valid_trace_kwargs(status=status))
            assert trace.status == status

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            Trace(**_valid_trace_kwargs(status="not_a_status"))

    def test_status_required(self):
        with pytest.raises(ValidationError):
            Trace(spans=[], final_output=None)

    def test_trace_holds_spans(self):
        span = Span(
            step="ingestion",
            input='{"path": "doc.md"}',
            output='{"doc_id": "abc"}',
            latency_ms=3.0,
        )
        trace = Trace(**_valid_trace_kwargs(spans=[span], final_output="The answer is 42."))
        assert trace.spans == [span]
        assert trace.final_output == "The answer is 42."

    def test_round_trip_json_with_spans(self):
        span = Span(
            step="generation",
            input='{"prompt": "..."}',
            output="The answer is 42.",
            llm_prompt="Answer using only the provided context.",
            token_count=300,
            latency_ms=850.2,
            confidence_score=4,
        )
        trace = Trace(
            spans=[span], final_output="The answer is 42.", status="success"
        )
        restored = Trace.model_validate_json(trace.model_dump_json())
        assert restored == trace

    def test_round_trip_model_dump(self):
        span = Span(step="ranking", input="[]", output="[]", latency_ms=5.0)
        trace = Trace(spans=[span], status="degraded")
        restored = Trace.model_validate(trace.model_dump())
        assert restored == trace
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/tracing/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'Trace' from 'src.tracing.models'`

- [ ] **Step 3: Implement `Trace` in `src/tracing/models.py`**

Append to `src/tracing/models.py`:

```python
TraceStatus = Literal["success", "failure", "degraded"]


class Trace(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    spans: list[Span] = Field(default_factory=list)
    final_output: str | None = None
    status: TraceStatus
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/tracing/test_models.py -v`
Expected: PASS (all `TestSpanValidation` and `TestTraceValidation` tests green)

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `pytest -q`
Expected: PASS, no failures elsewhere

- [ ] **Step 6: Lint and type-check**

Run: `ruff check src/tracing/ tests/unit/tracing/ && ruff format --check src/tracing/ tests/unit/tracing/ && mypy src/tracing/`
Expected: no errors (fix formatting with `ruff format src/tracing/ tests/unit/tracing/` if needed, then re-run)

- [ ] **Step 7: Commit**

```bash
git add src/tracing/models.py tests/unit/tracing/test_models.py
git commit -m "feat(tracing): add Trace pydantic model"
```
