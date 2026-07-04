# Span Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `span()` context manager + `traced()` decorator in `src/tracing/`, then instrument every existing retrieval and generation pipeline function with them, so each records a `Span` (step, serialized input/output, LLM prompt/response/token count where applicable, latency, errors).

**Architecture:** `src/tracing/context.py` holds a `ContextVar`-based `collect_spans()` sink; outside an active `collect_spans()` block, instrumentation is a complete no-op (existing tests that call pipeline functions directly, with no tracing setup, keep passing unmodified). `src/tracing/instrumentation.py` provides `span(step, input)` (a context manager for sites needing to attach LLM prompt/token detail mid-function) and `traced(step)` (a decorator built on `span()` that auto-serializes a function's arguments and return value — "one line of code" per the task). Both append a completed `Span` to whichever list `collect_spans()` currently has active, if any. No orchestrator is built here — see `docs/superpowers/specs/2026-07-04-span-instrumentation-design.md` for why that's out of scope.

**Tech Stack:** Python 3.11+, pydantic 2.x, pytest, `unittest.mock`. No new dependencies.

## Global Constraints

- Python `>=3.11` (per `pyproject.toml`) — use `from __future__ import annotations` in new modules to match existing style.
- `ruff check src/ tests/` and `ruff format src/ tests/` must pass on touched files only (do not reformat unrelated files — see repo convention on pre-existing ruff-format drift).
- `mypy src/` must pass.
- Every touched module keeps its existing lazy-import-for-optional-SDK pattern (`anthropic`/`openai`/`cohere`/`voyageai`/`sentence_transformers` imported inside `__init__`, not at module top level) — instrumentation must not add a new top-level import of any of those.
- No new environment variables or `Settings` fields — this task adds no configuration surface.
- Commit after every task.

---

### Task 1: `Span.error` field

**Files:**
- Modify: `src/tracing/models.py`
- Test: `tests/unit/tracing/test_models.py`

**Interfaces:**
- Produces: `Span.error: str | None = None` — every later task's `span()`/`traced()` implementation sets this on exception.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/tracing/test_models.py`, inside `TestSpanValidation`:

```python
    def test_error_defaults_to_none(self):
        span = Span(**_valid_span_kwargs())
        assert span.error is None

    def test_error_field_populated(self):
        span = Span(**_valid_span_kwargs(error="RuntimeError: boom"))
        assert span.error == "RuntimeError: boom"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/tracing/test_models.py -k error -v`
Expected: FAIL — `TypeError: Span() got an unexpected keyword argument 'error'` (or pydantic `ValidationError` on `test_error_field_populated`; `test_error_defaults_to_none` fails with `AttributeError: 'Span' object has no attribute 'error'`).

- [ ] **Step 3: Add the field**

In `src/tracing/models.py`, add to `Span` right after `confidence_score`:

```python
    confidence_score: int | None = Field(default=None, ge=1, le=5)
    error: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/tracing/test_models.py -v`
Expected: PASS (all tests, including the two new ones and every pre-existing one — `error` defaults to `None` so `test_round_trip_json`/`test_round_trip_json_with_spans` etc. are unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/tracing/models.py tests/unit/tracing/test_models.py
git commit -m "feat(tracing): add error field to Span"
```

---

### Task 2: `collect_spans()` span-collection sink

**Files:**
- Create: `src/tracing/context.py`
- Test: `tests/unit/tracing/test_context.py`

**Interfaces:**
- Consumes: `Span` from `src.tracing.models`.
- Produces: `collect_spans() -> ContextManager[list[Span]]` and `_active_sink() -> list[Span] | None` — both imported by `src/tracing/instrumentation.py` in Task 3.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/tracing/test_context.py`:

```python
from __future__ import annotations

from src.tracing.context import _active_sink, collect_spans
from src.tracing.models import Span


def _span(step: str = "retrieval") -> Span:
    return Span(step=step, input="in", output="out", latency_ms=1.0)


class TestCollectSpans:
    def test_no_active_sink_outside_context(self):
        assert _active_sink() is None

    def test_yields_empty_list_initially(self):
        with collect_spans() as spans:
            assert spans == []

    def test_appended_spans_visible_through_yielded_list(self):
        with collect_spans() as spans:
            spans.append(_span())
            assert len(spans) == 1

    def test_active_sink_is_the_yielded_list(self):
        with collect_spans() as spans:
            sink = _active_sink()
            assert sink is spans

    def test_sink_resets_after_exit(self):
        with collect_spans():
            pass
        assert _active_sink() is None

    def test_sink_resets_after_exception(self):
        try:
            with collect_spans():
                raise ValueError("boom")
        except ValueError:
            pass
        assert _active_sink() is None

    def test_nested_collect_spans_restores_outer_sink(self):
        with collect_spans() as outer:
            outer.append(_span("ingestion"))
            with collect_spans() as inner:
                inner.append(_span("retrieval"))
                assert _active_sink() is inner
            assert _active_sink() is outer
            assert len(outer) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/tracing/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tracing.context'`.

- [ ] **Step 3: Write the implementation**

Create `src/tracing/context.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from src.tracing.models import Span

_current_spans: ContextVar[list[Span] | None] = ContextVar(
    "_current_spans", default=None
)


@contextmanager
def collect_spans() -> Iterator[list[Span]]:
    """Activate a span-collection sink for the duration of the block.

    Instrumented calls (`span()`/`traced()` in `src.tracing.instrumentation`)
    append their completed `Span` to the returned list if this context is
    active when they run. Outside any `collect_spans()` block, instrumented
    calls have nowhere to append to and are a no-op with respect to tracing.
    """
    spans: list[Span] = []
    token = _current_spans.set(spans)
    try:
        yield spans
    finally:
        _current_spans.reset(token)


def _active_sink() -> list[Span] | None:
    """Return the currently active span list, or None if untraced."""
    return _current_spans.get()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/tracing/test_context.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tracing/context.py tests/unit/tracing/test_context.py
git commit -m "feat(tracing): add collect_spans contextvar-based span sink"
```

---

### Task 3: `span()` context manager + `default_serialize()`

**Files:**
- Create: `src/tracing/instrumentation.py`
- Test: `tests/unit/tracing/test_instrumentation.py`

**Interfaces:**
- Consumes: `collect_spans`/`_active_sink` from `src.tracing.context`; `Span`, `PipelineStep` from `src.tracing.models`.
- Produces: `span(step: PipelineStep, input: str) -> ContextManager[_SpanBuilder]`, `_SpanBuilder` (mutable, fields `step`, `input`, `output: str = ""`, `llm_prompt: str | None = None`, `token_count: int | None = None`, `confidence_score: int | None = None`), `default_serialize(value: object) -> str`. All consumed by Task 4 (`traced()`) and every later task's provider files.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/tracing/test_instrumentation.py`:

```python
from __future__ import annotations

import time

import pytest

from src.tracing.context import collect_spans
from src.tracing.instrumentation import default_serialize, span


class TestSpanContextManager:
    def test_no_active_sink_is_noop(self):
        with span("retrieval", input="q") as s:
            s.output = "result"
        # No assertion target — just must not raise.

    def test_records_span_into_active_sink(self):
        with collect_spans() as spans:
            with span("retrieval", input="q") as s:
                s.output = "result"

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "retrieval"
        assert recorded.input == "q"
        assert recorded.output == "result"
        assert recorded.error is None

    def test_measures_latency(self):
        with collect_spans() as spans:
            with span("retrieval", input="q") as s:
                time.sleep(0.01)
                s.output = "result"

        assert spans[0].latency_ms >= 10.0

    def test_captures_llm_prompt_and_token_count(self):
        with collect_spans() as spans:
            with span("verification", input="claim") as s:
                s.llm_prompt = "system + user prompt"
                s.token_count = 42
                s.output = "verdict"

        assert spans[0].llm_prompt == "system + user prompt"
        assert spans[0].token_count == 42

    def test_exception_is_recorded_and_reraised(self):
        with collect_spans() as spans:
            with pytest.raises(RuntimeError, match="boom"):
                with span("generation", input="q"):
                    raise RuntimeError("boom")

        assert len(spans) == 1
        assert spans[0].error == "RuntimeError: boom"
        assert spans[0].output == ""

    def test_no_sink_still_reraises_exception(self):
        with pytest.raises(RuntimeError, match="boom"):
            with span("generation", input="q"):
                raise RuntimeError("boom")

    def test_multiple_spans_append_in_order(self):
        with collect_spans() as spans:
            with span("retrieval", input="a") as s:
                s.output = "1"
            with span("ranking", input="b") as s:
                s.output = "2"

        assert [s.step for s in spans] == ["retrieval", "ranking"]


class TestDefaultSerialize:
    def test_serializes_plain_dict(self):
        result = default_serialize({"query": "q", "k": 5})
        assert result == '{"query": "q", "k": 5}'

    def test_serializes_dataclass(self):
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int
            y: int

        result = default_serialize(Point(1, 2))
        assert result == '{"x": 1, "y": 2}'

    def test_serializes_pydantic_model(self):
        from pydantic import BaseModel

        class Verdict(BaseModel):
            supported: bool
            reasoning: str

        result = default_serialize(Verdict(supported=True, reasoning="ok"))
        assert result == '{"supported": true, "reasoning": "ok"}'

    def test_serializes_list_of_dataclasses(self):
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int

        result = default_serialize([Point(1), Point(2)])
        assert result == '[{"x": 1}, {"x": 2}]'

    def test_falls_back_to_repr_for_unsupported_type(self):
        class Weird:
            def __repr__(self) -> str:
                return "Weird()"

        result = default_serialize(Weird())
        assert result == '"Weird()"'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/tracing/test_instrumentation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tracing.instrumentation'`.

- [ ] **Step 3: Write the implementation**

Create `src/tracing/instrumentation.py`:

```python
from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from pydantic import BaseModel

from src.tracing.context import _active_sink
from src.tracing.models import PipelineStep, Span


@dataclass
class _SpanBuilder:
    """Mutable in-progress span, yielded by `span()` for the caller to fill in."""

    step: PipelineStep
    input: str
    output: str = ""
    llm_prompt: str | None = None
    token_count: int | None = None
    confidence_score: int | None = None


def _serialize_item(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, list):
        return [_serialize_item(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_item(v) for k, v in value.items()}
    return value


def default_serialize(value: object) -> str:
    """Best-effort JSON serialization for span input/output.

    Not a strict schema — good enough for a human reading a trace, not meant
    to round-trip back into the original Python objects. Handles pydantic
    models, dataclasses, and lists/dicts of either; anything else falls back
    to `repr()`.
    """
    return json.dumps(_serialize_item(value), default=repr)


@contextmanager
def span(step: PipelineStep, input: str) -> Iterator[_SpanBuilder]:
    """Time a pipeline step and record it as a `Span` if tracing is active.

    Yields a mutable `_SpanBuilder` the caller fills in (`output`, and
    optionally `llm_prompt`/`token_count`/`confidence_score`) during the
    block. On exception, the span's `error` is set to
    `f"{type(exc).__name__}: {exc}"` and the exception is re-raised
    unchanged — this context manager never swallows errors.

    If no `collect_spans()` block is currently active, the span is still
    timed but discarded (nothing to append to) — safe to use in code paths
    that run without tracing.
    """
    builder = _SpanBuilder(step=step, input=input)
    start = time.perf_counter()
    error: str | None = None
    try:
        yield builder
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sink = _active_sink()
        if sink is not None:
            latency_ms = (time.perf_counter() - start) * 1000
            sink.append(
                Span(
                    step=builder.step,
                    input=builder.input,
                    output=builder.output,
                    llm_prompt=builder.llm_prompt,
                    token_count=builder.token_count,
                    latency_ms=latency_ms,
                    confidence_score=builder.confidence_score,
                    error=error,
                )
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/tracing/test_instrumentation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tracing/instrumentation.py tests/unit/tracing/test_instrumentation.py
git commit -m "feat(tracing): add span() context manager and default_serialize"
```

---

### Task 4: `traced()` decorator

**Files:**
- Modify: `src/tracing/instrumentation.py`
- Test: `tests/unit/tracing/test_instrumentation.py`

**Interfaces:**
- Consumes: `span()`, `default_serialize()` (this module, Task 3).
- Produces: `traced(step: PipelineStep) -> Callable[[Callable], Callable]` — applied as `@traced("retrieval")` etc. in Tasks 5–15.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/tracing/test_instrumentation.py`:

```python
from src.tracing.instrumentation import traced


class TestTracedDecorator:
    def test_preserves_return_value(self):
        @traced("retrieval")
        def retrieve(query: str) -> list[str]:
            return [query, query]

        assert retrieve("q") == ["q", "q"]

    def test_records_span_with_serialized_args_and_result(self):
        @traced("retrieval")
        def retrieve(query: str, k: int = 10) -> list[str]:
            return [query] * k

        with collect_spans() as spans:
            retrieve("q", k=2)

        assert len(spans) == 1
        assert spans[0].step == "retrieval"
        assert '"query": "q"' in spans[0].input
        assert '"k": 2' in spans[0].input
        assert spans[0].output == '["q", "q"]'

    def test_default_argument_captured_even_when_omitted(self):
        @traced("retrieval")
        def retrieve(query: str, k: int = 10) -> list[str]:
            return [query] * k

        with collect_spans() as spans:
            retrieve("q")

        assert '"k": 10' in spans[0].input

    def test_excludes_self_from_serialized_input(self):
        class Retriever:
            @traced("retrieval")
            def retrieve(self, query: str) -> str:
                return query

        with collect_spans() as spans:
            Retriever().retrieve("q")

        assert "self" not in spans[0].input
        assert '"query": "q"' in spans[0].input

    def test_propagates_exception_and_still_records_span(self):
        @traced("generation")
        def flaky() -> None:
            raise ValueError("nope")

        with collect_spans() as spans:
            with pytest.raises(ValueError, match="nope"):
                flaky()

        assert len(spans) == 1
        assert spans[0].error == "ValueError: nope"

    def test_noop_outside_collect_spans(self):
        @traced("retrieval")
        def retrieve(query: str) -> str:
            return query

        assert retrieve("q") == "q"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/tracing/test_instrumentation.py -k Traced -v`
Expected: FAIL with `ImportError: cannot import name 'traced'`.

- [ ] **Step 3: Write the implementation**

Append to `src/tracing/instrumentation.py` (add `functools` and `inspect` to the imports at the top):

```python
import functools
import inspect
```

Then add at the end of the file:

```python
def traced(step: PipelineStep):
    """Decorator: wrap a function/method so each call records a `Span`.

    Auto-serializes the call's bound arguments (`self` excluded, defaults
    applied) as `input` via `default_serialize`, and the return value as
    `output`. For sites needing to attach LLM prompt/response/token detail
    that isn't derivable from arguments/return value alone, use `span()`
    directly instead.
    """

    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            arguments.pop("self", None)
            with span(step, input=default_serialize(arguments)) as s:
                result = func(*args, **kwargs)
                s.output = default_serialize(result)
                return result

        return wrapper

    return decorator
```

(`sig` is computed once at decoration time, not on every call — `wrapper`
closes over it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/tracing/test_instrumentation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tracing/instrumentation.py tests/unit/tracing/test_instrumentation.py
git commit -m "feat(tracing): add traced() decorator"
```

---

### Task 5: Instrument `DenseRetriever.retrieve`

**Files:**
- Modify: `src/retrieval/dense_retriever.py`
- Test: `tests/unit/retrieval/test_dense_retriever.py`

**Interfaces:**
- Consumes: `traced` from `src.tracing.instrumentation`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/retrieval/test_dense_retriever.py`:

```python
from src.tracing.context import collect_spans


class TestDenseRetrieverTracing:
    def test_retrieve_records_retrieval_span(self):
        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = [[1.0, 0.0, 0.0]]
        vs = MagicMock(spec=VectorStore)
        vs.query.return_value = [_hit(chunk_id="c-000")]

        with collect_spans() as spans:
            DenseRetriever(embedder, vs).retrieve("q")

        assert len(spans) == 1
        assert spans[0].step == "retrieval"
        assert spans[0].error is None

    def test_retrieve_noop_outside_collect_spans(self):
        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = [[1.0, 0.0, 0.0]]
        vs = MagicMock(spec=VectorStore)
        vs.query.return_value = []

        # Must not raise even with no active tracing context.
        DenseRetriever(embedder, vs).retrieve("q")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_dense_retriever.py -k Tracing -v`
Expected: FAIL — `assert len(spans) == 1` fails with `0 == 1` (no instrumentation yet).

- [ ] **Step 3: Add the decorator**

In `src/retrieval/dense_retriever.py`:

```python
from src.retrieval.embedder import EmbedderProtocol
from src.retrieval.models import VectorStoreHit
from src.retrieval.vector_store import VectorStoreProtocol
from src.tracing.instrumentation import traced

_DEFAULT_K = 10


class DenseRetriever:
    def __init__(self, embedder: EmbedderProtocol, vector_store: VectorStoreProtocol) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    @traced("retrieval")
    def retrieve(self, query: str, k: int = _DEFAULT_K) -> list[VectorStoreHit]:
        (embedding,) = self._embedder.embed([query])
        return self._vector_store.query(embedding, k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/retrieval/test_dense_retriever.py -v`
Expected: PASS (all tests, including pre-existing ones — the decorator preserves the return value and call signature).

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/dense_retriever.py tests/unit/retrieval/test_dense_retriever.py
git commit -m "feat(retrieval): instrument DenseRetriever.retrieve with a retrieval span"
```

---

### Task 6: Instrument `SparseRetriever.retrieve`

**Files:**
- Modify: `src/retrieval/sparse_retriever.py`
- Test: `tests/unit/retrieval/test_sparse_retriever.py`

**Interfaces:**
- Consumes: `traced` from `src.tracing.instrumentation`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/retrieval/test_sparse_retriever.py`:

```python
from src.tracing.context import collect_spans


class TestSparseRetrieverTracing:
    def test_retrieve_records_retrieval_span(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = [("c-000", 3.5)]
        vs = MagicMock(spec=VectorStore)
        vs.get_by_ids.return_value = [_hit(chunk_id="c-000")]

        with collect_spans() as spans:
            SparseRetriever(bm25, vs).retrieve("q")

        assert len(spans) == 1
        assert spans[0].step == "retrieval"
        assert spans[0].error is None

    def test_retrieve_noop_outside_collect_spans(self):
        bm25 = MagicMock(spec=BM25Store)
        bm25.get_scores.return_value = []
        vs = MagicMock(spec=VectorStore)

        SparseRetriever(bm25, vs).retrieve("q")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_sparse_retriever.py -k Tracing -v`
Expected: FAIL — `0 == 1`.

- [ ] **Step 3: Add the decorator**

In `src/retrieval/sparse_retriever.py`:

```python
import dataclasses

from src.retrieval.bm25_store import BM25Store
from src.retrieval.models import VectorStoreHit
from src.retrieval.vector_store import VectorStoreProtocol
from src.tracing.instrumentation import traced

_DEFAULT_K = 10


class SparseRetriever:
    def __init__(self, bm25_store: BM25Store, vector_store: VectorStoreProtocol) -> None:
        self._bm25_store = bm25_store
        self._vector_store = vector_store

    @traced("retrieval")
    def retrieve(self, query: str, k: int = _DEFAULT_K) -> list[VectorStoreHit]:
        scores = self._bm25_store.get_scores(query)
        if not scores:
            return []
        top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        top_k = [(cid, s) for cid, s in top_k if s > 0.0]
        if not top_k:
            return []
        max_score = top_k[0][1]
        top_k = [(cid, s / max_score) for cid, s in top_k]
        ids = [cid for cid, _ in top_k]
        score_map = {cid: s for cid, s in top_k}
        hits = self._vector_store.get_by_ids(ids)
        hit_map = {h.chunk_id: h for h in hits}
        return [
            dataclasses.replace(hit_map[cid], similarity=score_map[cid])
            for cid in ids
            if cid in hit_map
        ]
```

(Only the import block and the `@traced("retrieval")` line above `def retrieve` change — the method body is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/retrieval/test_sparse_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/sparse_retriever.py tests/unit/retrieval/test_sparse_retriever.py
git commit -m "feat(retrieval): instrument SparseRetriever.retrieve with a retrieval span"
```

---

### Task 7: Instrument `reciprocal_rank_fusion`

**Files:**
- Modify: `src/retrieval/fusion.py`
- Test: `tests/unit/retrieval/test_fusion.py`

**Interfaces:**
- Consumes: `traced` from `src.tracing.instrumentation`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/retrieval/test_fusion.py`:

```python
from src.tracing.context import collect_spans


class TestFusionTracing:
    def test_records_retrieval_span(self):
        hits = [_hit("c1")]

        with collect_spans() as spans:
            reciprocal_rank_fusion(hits, [], top_n=1)

        assert len(spans) == 1
        assert spans[0].step == "retrieval"
        assert spans[0].error is None

    def test_noop_outside_collect_spans(self):
        reciprocal_rank_fusion([], [], top_n=5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/test_fusion.py -k Tracing -v`
Expected: FAIL — `0 == 1`.

- [ ] **Step 3: Add the decorator**

In `src/retrieval/fusion.py`:

```python
from dataclasses import replace

from src.retrieval.models import VectorStoreHit
from src.tracing.instrumentation import traced

_RRF_K = 60


@traced("retrieval")
def reciprocal_rank_fusion(
    dense_hits: list[VectorStoreHit],
    sparse_hits: list[VectorStoreHit],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    top_n: int = 5,
) -> list[VectorStoreHit]:
    scores: dict[str, float] = {}
    hits_by_id: dict[str, VectorStoreHit] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + dense_weight / (
            _RRF_K + rank
        )
        hits_by_id[hit.chunk_id] = hit

    for rank, hit in enumerate(sparse_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + sparse_weight / (
            _RRF_K + rank
        )
        hits_by_id.setdefault(hit.chunk_id, hit)

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_n]
    return [replace(hits_by_id[cid], similarity=scores[cid]) for cid in sorted_ids]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/retrieval/test_fusion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/fusion.py tests/unit/retrieval/test_fusion.py
git commit -m "feat(retrieval): instrument reciprocal_rank_fusion with a retrieval span"
```

---

### Task 8: Instrument `SentenceTransformersReranker.rerank`

**Files:**
- Modify: `src/retrieval/providers/reranker_sentence_transformers.py`
- Test: `tests/unit/retrieval/providers/test_reranker_sentence_transformers.py`

**Interfaces:**
- Consumes: `traced` from `src.tracing.instrumentation`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/retrieval/providers/test_reranker_sentence_transformers.py`:

```python
    def test_rerank_records_ranking_span(self):
        from src.retrieval.providers.reranker_sentence_transformers import (
            SentenceTransformersReranker,
        )
        from src.tracing.context import collect_spans

        hits = [_hit("a")]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = SentenceTransformersReranker()
            with collect_spans() as spans:
                reranker.rerank("q", hits, top_n=1)

        assert len(spans) == 1
        assert spans[0].step == "ranking"
        assert spans[0].error is None
```

Add this method inside `class TestSentenceTransformersReranker:`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/providers/test_reranker_sentence_transformers.py -k ranking_span -v`
Expected: FAIL — `0 == 1`.

- [ ] **Step 3: Add the decorator**

In `src/retrieval/providers/reranker_sentence_transformers.py`, add the import and decorate `rerank`:

```python
from src.retrieval.models import VectorStoreHit
from src.tracing.instrumentation import traced

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


class SentenceTransformersReranker:
    ...

    @traced("ranking")
    def rerank(
        self, query: str, hits: list[VectorStoreHit], top_n: int
    ) -> list[VectorStoreHit]:
        """Re-score *hits* against *query* and return the top_n, sorted descending."""
        if not hits:
            return []
        pairs = [(query, hit.text) for hit in hits]
        scores = self._model.predict(pairs)
        scored = sorted(
            zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [replace(hit, similarity=float(score)) for hit, score in scored[:top_n]]
```

(Only the import block and the `@traced("ranking")` line change — everything else in the file is unchanged, including `__init__` and `provider_id`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/retrieval/providers/test_reranker_sentence_transformers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/providers/reranker_sentence_transformers.py tests/unit/retrieval/providers/test_reranker_sentence_transformers.py
git commit -m "feat(retrieval): instrument SentenceTransformersReranker.rerank with a ranking span"
```

---

### Task 9: Instrument `CohereReranker.rerank`

**Files:**
- Modify: `src/retrieval/providers/reranker_cohere.py`
- Test: `tests/unit/retrieval/providers/test_reranker_cohere.py`

**Interfaces:**
- Consumes: `traced` from `src.tracing.instrumentation`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/retrieval/providers/test_reranker_cohere.py` (as a new test class, using the file's existing `settings` fixture and `_hit`/`_result`/`_mock_response` helpers):

```python
class TestCohereRerankerTracing:
    def test_rerank_records_ranking_span(self, settings):
        from src.retrieval.providers.reranker_cohere import CohereReranker
        from src.tracing.context import collect_spans

        hits = [_hit("a")]
        with patch("cohere.ClientV2") as MockClient:
            MockClient.return_value.v2.rerank.return_value = _mock_response(
                [_result(0, 0.9)]
            )
            reranker = CohereReranker(settings)
            with collect_spans() as spans:
                reranker.rerank("q", hits, top_n=1)

        assert len(spans) == 1
        assert spans[0].step == "ranking"
        assert spans[0].error is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/providers/test_reranker_cohere.py -k Tracing -v`
Expected: FAIL — `0 == 1`.

- [ ] **Step 3: Add the decorator**

In `src/retrieval/providers/reranker_cohere.py`:

```python
from src.retrieval.models import VectorStoreHit
from src.tracing.instrumentation import traced

if TYPE_CHECKING:
    from src.config import Settings

DEFAULT_MODEL = "rerank-v4.0-pro"


class CohereReranker:
    ...

    @traced("ranking")
    def rerank(
        self, query: str, hits: list[VectorStoreHit], top_n: int
    ) -> list[VectorStoreHit]:
        """Re-score *hits* against *query* and return the top_n, sorted descending."""
        if not hits:
            return []
        response = self._client.v2.rerank(
            model=self._model,
            query=query,
            documents=[hit.text for hit in hits],
            top_n=top_n,
        )
        return [
            replace(hits[result.index], similarity=float(result.relevance_score))
            for result in response.results
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/retrieval/providers/test_reranker_cohere.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/providers/reranker_cohere.py tests/unit/retrieval/providers/test_reranker_cohere.py
git commit -m "feat(retrieval): instrument CohereReranker.rerank with a ranking span"
```

---

### Task 10: Instrument `VoyageReranker.rerank`

**Files:**
- Modify: `src/retrieval/providers/reranker_voyage.py`
- Test: `tests/unit/retrieval/providers/test_reranker_voyage.py`

**Interfaces:**
- Consumes: `traced` from `src.tracing.instrumentation`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/retrieval/providers/test_reranker_voyage.py`:

```python
class TestVoyageRerankerTracing:
    def test_rerank_records_ranking_span(self, settings):
        from src.retrieval.providers.reranker_voyage import VoyageReranker
        from src.tracing.context import collect_spans

        hits = [_hit("a")]
        with patch("voyageai.Client") as MockClient:
            MockClient.return_value.rerank.return_value = _mock_response(
                [_result(0, 0.9)]
            )
            reranker = VoyageReranker(settings)
            with collect_spans() as spans:
                reranker.rerank("q", hits, top_n=1)

        assert len(spans) == 1
        assert spans[0].step == "ranking"
        assert spans[0].error is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/retrieval/providers/test_reranker_voyage.py -k Tracing -v`
Expected: FAIL — `0 == 1`.

- [ ] **Step 3: Add the decorator**

In `src/retrieval/providers/reranker_voyage.py`:

```python
from src.retrieval.models import VectorStoreHit
from src.tracing.instrumentation import traced

if TYPE_CHECKING:
    from src.config import Settings

DEFAULT_MODEL = "rerank-2.5"


class VoyageReranker:
    ...

    @traced("ranking")
    def rerank(
        self, query: str, hits: list[VectorStoreHit], top_n: int
    ) -> list[VectorStoreHit]:
        """Re-score *hits* against *query* and return the top_n, sorted descending."""
        if not hits:
            return []
        result = self._client.rerank(
            query,
            [hit.text for hit in hits],
            model=self._model,
            top_k=top_n,
        )
        return [
            replace(hits[r.index], similarity=float(r.relevance_score))
            for r in result.results
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/retrieval/providers/test_reranker_voyage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/providers/reranker_voyage.py tests/unit/retrieval/providers/test_reranker_voyage.py
git commit -m "feat(retrieval): instrument VoyageReranker.rerank with a ranking span"
```

---

### Task 11: Instrument `verify_citations`

**Files:**
- Modify: `src/generation/citation_verifier.py`
- Test: `tests/unit/generation/test_citation_verifier.py`

**Interfaces:**
- Consumes: `traced` from `src.tracing.instrumentation`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/generation/test_citation_verifier.py`, inside `class TestVerifyCitations:`:

```python
    def test_records_verification_span(self):
        from src.tracing.context import collect_spans

        hits = [make_hit(chunk_id="a", text="Paris is the capital of France.")]
        judge = FakeJudge()
        answer = "Paris is the capital [1]."

        with collect_spans() as spans:
            verify_citations(answer, hits, judge)

        assert len(spans) == 1
        assert spans[0].step == "verification"
        assert spans[0].error is None

    def test_noop_outside_collect_spans(self):
        hits = [make_hit()]
        judge = FakeJudge()

        verify_citations("Some claim [1].", hits, judge)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/generation/test_citation_verifier.py -k "records_verification_span or noop_outside" -v`
Expected: FAIL — `0 == 1`.

- [ ] **Step 3: Add the decorator**

In `src/generation/citation_verifier.py`, add the import and decorate `verify_citations`:

```python
from src.generation.citation_parser import parse_citations
from src.generation.prompts import GroundedPrompt, wrap_with_nonce
from src.retrieval.models import VectorStoreHit
from src.tracing.instrumentation import traced

if TYPE_CHECKING:
    from src.config import Settings
```

```python
@traced("verification")
def verify_citations(
    answer_text: str, hits: list[VectorStoreHit], judge: CitationJudgeProtocol
) -> list[CitationVerificationResult]:
    """Verify every citation in *answer_text* against the retrieved *hits*.
    ...
    """
    if not hits:
        return []
    # ... rest of the function body is unchanged
```

(Keep the full existing docstring and body — only the import block and the
new `@traced("verification")` decorator line are added.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/generation/test_citation_verifier.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/generation/citation_verifier.py tests/unit/generation/test_citation_verifier.py
git commit -m "feat(generation): instrument verify_citations with a verification span"
```

---

### Task 12: Instrument citation judge providers (Anthropic + OpenAI)

**Files:**
- Modify: `src/generation/providers/citation_judge_anthropic.py`
- Modify: `src/generation/providers/citation_judge_openai.py`
- Test: `tests/unit/generation/providers/test_citation_judge_anthropic.py`
- Test: `tests/unit/generation/providers/test_citation_judge_openai.py`

**Interfaces:**
- Consumes: `span`, `default_serialize` from `src.tracing.instrumentation`.
- Token extraction confirmed via Context7 against the installed SDKs: Anthropic's `Message.usage` has `input_tokens: int` and `output_tokens: int` (`anthropic/types/usage.py`); OpenAI's `ChatCompletion.usage` is `Optional[CompletionUsage]` with a `total_tokens: int` field. Both provider files use `isinstance(..., int)` checks (not exception handling) when extracting token counts, because `unittest.mock.MagicMock` auto-vivifies any attribute access (including `.usage.input_tokens`) rather than raising `AttributeError` — an `isinstance` check is the only reliable way to detect "this test didn't set real usage data" and fall back to `None` instead of leaking a `MagicMock` into `Span.token_count` (which would fail pydantic validation, breaking every pre-existing test that doesn't set `.usage`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/generation/providers/test_citation_judge_anthropic.py`, inside `class TestAnthropicCitationJudge:`:

```python
    def test_judge_records_span_with_prompt_and_token_count(self, settings):
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )
        from src.tracing.context import collect_spans

        resp = _mock_response(supported=True, reasoning="Matches.")
        resp.usage.input_tokens = 120
        resp.usage.output_tokens = 30

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = resp
            judge = AnthropicCitationJudge(settings)
            with collect_spans() as spans:
                judge.judge(claim="The sky is blue.", evidence="The sky appears blue.")

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "verification"
        assert recorded.token_count == 150
        assert "The sky is blue." in recorded.llm_prompt
        assert recorded.error is None

    def test_judge_noop_outside_collect_spans(self, settings):
        from src.generation.providers.citation_judge_anthropic import (
            AnthropicCitationJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response()
            judge = AnthropicCitationJudge(settings)
            judge.judge(claim="c", evidence="e")
```

Add to `tests/unit/generation/providers/test_citation_judge_openai.py`, inside `class TestOpenAICitationJudge:`:

```python
    def test_judge_records_span_with_prompt_and_token_count(self, settings):
        from src.generation.providers.citation_judge_openai import OpenAICitationJudge
        from src.tracing.context import collect_spans

        completion = _mock_completion(supported=True, reasoning="Matches.")
        completion.usage.total_tokens = 200

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = completion
            judge = OpenAICitationJudge(settings)
            with collect_spans() as spans:
                judge.judge(claim="The sky is blue.", evidence="The sky appears blue.")

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "verification"
        assert recorded.token_count == 200
        assert "The sky is blue." in recorded.llm_prompt
        assert recorded.error is None

    def test_judge_noop_outside_collect_spans(self, settings):
        from src.generation.providers.citation_judge_openai import OpenAICitationJudge

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion()
            )
            judge = OpenAICitationJudge(settings)
            judge.judge(claim="c", evidence="e")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/generation/providers/test_citation_judge_anthropic.py tests/unit/generation/providers/test_citation_judge_openai.py -k token_count -v`
Expected: FAIL — `assert len(spans) == 1` fails with `0 == 1` (no instrumentation yet).

- [ ] **Step 3: Instrument both providers**

`src/generation/providers/citation_judge_anthropic.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from src.generation.citation_verifier import (
    CITATION_JUDGE_SYSTEM_PROMPT,
    JudgeVerdict,
    build_judge_prompt,
)
from src.tracing.instrumentation import default_serialize, span

if TYPE_CHECKING:
    from src.config import Settings

DEFAULT_MODEL = "claude-sonnet-4-5"


def _extract_token_count(response: object) -> int | None:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return input_tokens + output_tokens
    return None


class AnthropicCitationJudge:
    """Citation judge backed by the Anthropic Messages API structured output."""

    def __init__(self, settings: Settings) -> None:
        from anthropic import Anthropic  # lazy import — not at module level

        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.citation_judge_model
        self._temperature = settings.citation_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"anthropic/claude-sonnet-4-5"``."""
        return f"anthropic/{self._model}"

    def judge(self, claim: str, evidence: str) -> JudgeVerdict:
        """Decide whether *evidence* supports *claim* and return a verdict."""
        prompt = build_judge_prompt(claim, evidence)
        with span(
            "verification",
            input=default_serialize({"claim": claim, "evidence": evidence}),
        ) as s:
            s.llm_prompt = f"{CITATION_JUDGE_SYSTEM_PROMPT}\n\n{prompt.user}"
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=1024,
                system=CITATION_JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt.user}],
                temperature=self._temperature,
                output_format=JudgeVerdict,
            )
            s.token_count = _extract_token_count(response)
            parsed = response.parsed_output
            if parsed is None:
                raise RuntimeError(
                    f"Anthropic structured output returned no parsed_output "
                    f"(model={self._model})"
                )
            s.output = default_serialize(parsed)
            return parsed
```

`src/generation/providers/citation_judge_openai.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from src.generation.citation_verifier import (
    CITATION_JUDGE_SYSTEM_PROMPT,
    JudgeVerdict,
    build_judge_prompt,
)
from src.tracing.instrumentation import default_serialize, span

if TYPE_CHECKING:
    from src.config import Settings

DEFAULT_MODEL = "gpt-4o-2024-08-06"


def _extract_token_count(completion: object) -> int | None:
    usage = getattr(completion, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return total_tokens if isinstance(total_tokens, int) else None


class OpenAICitationJudge:
    """Citation judge backed by the OpenAI Chat Completions structured output."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # lazy import — not at module level

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.citation_judge_model
        self._temperature = settings.citation_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"openai/gpt-4o-2024-08-06"``."""
        return f"openai/{self._model}"

    def judge(self, claim: str, evidence: str) -> JudgeVerdict:
        """Decide whether *evidence* supports *claim* and return a verdict."""
        prompt = build_judge_prompt(claim, evidence)
        with span(
            "verification",
            input=default_serialize({"claim": claim, "evidence": evidence}),
        ) as s:
            s.llm_prompt = f"{CITATION_JUDGE_SYSTEM_PROMPT}\n\n{prompt.user}"
            completion = self._client.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": CITATION_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt.user},
                ],
                temperature=self._temperature,
                response_format=JudgeVerdict,
            )
            s.token_count = _extract_token_count(completion)
            message = completion.choices[0].message
            parsed = message.parsed
            if parsed is None:
                raise RuntimeError(
                    f"OpenAI structured output returned no parsed result "
                    f"(model={self._model}, refusal={message.refusal!r})"
                )
            s.output = default_serialize(parsed)
            return parsed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/generation/providers/test_citation_judge_anthropic.py tests/unit/generation/providers/test_citation_judge_openai.py -v`
Expected: PASS (all tests, including pre-existing ones — `_extract_token_count` returns `None` for the pre-existing tests' `MagicMock`-only `.usage`, which is a valid `Span.token_count` value).

- [ ] **Step 5: Commit**

```bash
git add src/generation/providers/citation_judge_anthropic.py src/generation/providers/citation_judge_openai.py tests/unit/generation/providers/test_citation_judge_anthropic.py tests/unit/generation/providers/test_citation_judge_openai.py
git commit -m "feat(generation): instrument citation judge providers with verification spans"
```

---

### Task 13: Instrument `score_confidence`

**Files:**
- Modify: `src/generation/confidence_scorer.py`
- Test: `tests/unit/generation/test_confidence_scorer.py`

**Interfaces:**
- Consumes: `traced` from `src.tracing.instrumentation`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/generation/test_confidence_scorer.py`, inside `class TestScoreConfidence:`:

```python
    def test_records_generation_span(self):
        from src.tracing.context import collect_spans

        judge = FakeCompletenessJudge()

        with collect_spans() as spans:
            score_confidence("q", "a", [make_hit()], [], judge)

        assert len(spans) == 1
        assert spans[0].step == "generation"
        assert spans[0].error is None

    def test_noop_outside_collect_spans(self):
        judge = FakeCompletenessJudge()

        score_confidence("q", "a", [], [], judge)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/generation/test_confidence_scorer.py -k "records_generation_span or noop_outside" -v`
Expected: FAIL — `0 == 1`.

- [ ] **Step 3: Add the decorator**

In `src/generation/confidence_scorer.py`, add the import:

```python
from src.generation.prompts import GroundedPrompt, wrap_with_nonce
from src.tracing.instrumentation import traced

if TYPE_CHECKING:
    from src.config import Settings
    from src.generation.citation_verifier import CitationVerificationResult
    from src.retrieval.models import VectorStoreHit
```

And decorate the function, keeping its existing docstring and body unchanged:

```python
@traced("generation")
def score_confidence(
    query: str,
    answer_text: str,
    hits: list[VectorStoreHit],
    citation_results: list[CitationVerificationResult],
    judge: CompletenessJudgeProtocol,
    retrieval_weight: float = 1 / 3,
    citation_weight: float = 1 / 3,
    completeness_weight: float = 1 / 3,
) -> ConfidenceScore:
    """Score a generated answer on retrieval, citation, and completeness.
    ...
    """
    # ... body unchanged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/generation/test_confidence_scorer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/generation/confidence_scorer.py tests/unit/generation/test_confidence_scorer.py
git commit -m "feat(generation): instrument score_confidence with a generation span"
```

---

### Task 14: Instrument completeness judge providers (Anthropic + OpenAI)

**Files:**
- Modify: `src/generation/providers/completeness_judge_anthropic.py`
- Modify: `src/generation/providers/completeness_judge_openai.py`
- Test: `tests/unit/generation/providers/test_completeness_judge_anthropic.py`
- Test: `tests/unit/generation/providers/test_completeness_judge_openai.py`

**Interfaces:**
- Consumes: `span`, `default_serialize` from `src.tracing.instrumentation`.
- Same `_extract_token_count` pattern and rationale as Task 12, mapped to `step="generation"` instead of `"verification"` (per the design spec: the completeness judge assesses the generated answer as a whole, which fits the `generation` bucket in the closed `PipelineStep` literal better than any of the other four values).

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/generation/providers/test_completeness_judge_anthropic.py`, inside `class TestAnthropicCompletenessJudge:`:

```python
    def test_judge_records_span_with_prompt_and_token_count(self, settings):
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )
        from src.tracing.context import collect_spans

        resp = _mock_response(complete=True, reasoning="Addresses both parts.")
        resp.usage.input_tokens = 80
        resp.usage.output_tokens = 20

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = resp
            judge = AnthropicCompletenessJudge(settings)
            with collect_spans() as spans:
                judge.judge(question="What is X?", answer="X is a thing.")

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "generation"
        assert recorded.token_count == 100
        assert "What is X?" in recorded.llm_prompt
        assert recorded.error is None

    def test_judge_noop_outside_collect_spans(self, settings):
        from src.generation.providers.completeness_judge_anthropic import (
            AnthropicCompletenessJudge,
        )

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.parse.return_value = _mock_response()
            judge = AnthropicCompletenessJudge(settings)
            judge.judge(question="q", answer="a")
```

Add to `tests/unit/generation/providers/test_completeness_judge_openai.py` (mirror the citation judge OpenAI test structure — check that file's existing `_mock_completion`/`settings` helpers before writing, they follow the same shape as `test_citation_judge_openai.py`):

```python
    def test_judge_records_span_with_prompt_and_token_count(self, settings):
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )
        from src.tracing.context import collect_spans

        completion = _mock_completion(complete=True, reasoning="Addresses both parts.")
        completion.usage.total_tokens = 150

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = completion
            judge = OpenAICompletenessJudge(settings)
            with collect_spans() as spans:
                judge.judge(question="What is X?", answer="X is a thing.")

        assert len(spans) == 1
        recorded = spans[0]
        assert recorded.step == "generation"
        assert recorded.token_count == 150
        assert "What is X?" in recorded.llm_prompt
        assert recorded.error is None

    def test_judge_noop_outside_collect_spans(self, settings):
        from src.generation.providers.completeness_judge_openai import (
            OpenAICompletenessJudge,
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.parse.return_value = (
                _mock_completion()
            )
            judge = OpenAICompletenessJudge(settings)
            judge.judge(question="q", answer="a")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/generation/providers/test_completeness_judge_anthropic.py tests/unit/generation/providers/test_completeness_judge_openai.py -k token_count -v`
Expected: FAIL — `0 == 1`.

- [ ] **Step 3: Instrument both providers**

`src/generation/providers/completeness_judge_anthropic.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from src.generation.confidence_scorer import (
    ANSWER_COMPLETENESS_SYSTEM_PROMPT,
    CompletenessVerdict,
    build_completeness_judge_prompt,
)
from src.tracing.instrumentation import default_serialize, span

if TYPE_CHECKING:
    from src.config import Settings

DEFAULT_MODEL = "claude-sonnet-4-5"


def _extract_token_count(response: object) -> int | None:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return input_tokens + output_tokens
    return None


class AnthropicCompletenessJudge:
    """Answer completeness judge backed by the Anthropic Messages API structured output."""

    def __init__(self, settings: Settings) -> None:
        from anthropic import Anthropic  # lazy import — not at module level

        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.answer_completeness_judge_model
        self._temperature = settings.answer_completeness_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"anthropic/claude-sonnet-4-5"``."""
        return f"anthropic/{self._model}"

    def judge(self, question: str, answer: str) -> CompletenessVerdict:
        """Decide whether *answer* addresses every part of *question*."""
        prompt = build_completeness_judge_prompt(question, answer)
        with span(
            "generation",
            input=default_serialize({"question": question, "answer": answer}),
        ) as s:
            s.llm_prompt = f"{ANSWER_COMPLETENESS_SYSTEM_PROMPT}\n\n{prompt.user}"
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=1024,
                system=ANSWER_COMPLETENESS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt.user}],
                temperature=self._temperature,
                output_format=CompletenessVerdict,
            )
            s.token_count = _extract_token_count(response)
            parsed = response.parsed_output
            if parsed is None:
                raise RuntimeError(
                    f"Anthropic structured output returned no parsed_output "
                    f"(model={self._model})"
                )
            s.output = default_serialize(parsed)
            return parsed
```

`src/generation/providers/completeness_judge_openai.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from src.generation.confidence_scorer import (
    ANSWER_COMPLETENESS_SYSTEM_PROMPT,
    CompletenessVerdict,
    build_completeness_judge_prompt,
)
from src.tracing.instrumentation import default_serialize, span

if TYPE_CHECKING:
    from src.config import Settings

DEFAULT_MODEL = "gpt-4o-2024-08-06"


def _extract_token_count(completion: object) -> int | None:
    usage = getattr(completion, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return total_tokens if isinstance(total_tokens, int) else None


class OpenAICompletenessJudge:
    """Answer completeness judge backed by the OpenAI Chat Completions structured output."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # lazy import — not at module level

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.answer_completeness_judge_model
        self._temperature = settings.answer_completeness_judge_temperature

    @property
    def provider_id(self) -> str:
        """Provider identifier including model name, e.g. ``"openai/gpt-4o-2024-08-06"``."""
        return f"openai/{self._model}"

    def judge(self, question: str, answer: str) -> CompletenessVerdict:
        """Decide whether *answer* addresses every part of *question*."""
        prompt = build_completeness_judge_prompt(question, answer)
        with span(
            "generation",
            input=default_serialize({"question": question, "answer": answer}),
        ) as s:
            s.llm_prompt = f"{ANSWER_COMPLETENESS_SYSTEM_PROMPT}\n\n{prompt.user}"
            completion = self._client.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": ANSWER_COMPLETENESS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt.user},
                ],
                temperature=self._temperature,
                response_format=CompletenessVerdict,
            )
            s.token_count = _extract_token_count(completion)
            message = completion.choices[0].message
            parsed = message.parsed
            if parsed is None:
                raise RuntimeError(
                    f"OpenAI structured output returned no parsed result "
                    f"(model={self._model}, refusal={message.refusal!r})"
                )
            s.output = default_serialize(parsed)
            return parsed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/generation/providers/test_completeness_judge_anthropic.py tests/unit/generation/providers/test_completeness_judge_openai.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/generation/providers/completeness_judge_anthropic.py src/generation/providers/completeness_judge_openai.py tests/unit/generation/providers/test_completeness_judge_anthropic.py tests/unit/generation/providers/test_completeness_judge_openai.py
git commit -m "feat(generation): instrument completeness judge providers with generation spans"
```

---

### Task 15: Instrument `build_fallback_response`

**Files:**
- Modify: `src/generation/fallback_response.py`
- Test: `tests/unit/generation/test_fallback_response.py`

**Interfaces:**
- Consumes: `traced` from `src.tracing.instrumentation`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/generation/test_fallback_response.py`, inside `class TestBuildFallbackResponse:`:

```python
    def test_records_generation_span(self):
        from src.tracing.context import collect_spans

        hits = [make_hit(similarity=0.2)]

        with collect_spans() as spans:
            build_fallback_response(hits, retrieval_confidence=0.2, threshold=0.5)

        assert len(spans) == 1
        assert spans[0].step == "generation"
        assert spans[0].error is None

    def test_records_span_even_when_returning_none(self):
        from src.tracing.context import collect_spans

        hits = [make_hit(similarity=0.9)]

        with collect_spans() as spans:
            result = build_fallback_response(hits, retrieval_confidence=0.9, threshold=0.5)

        assert result is None
        assert len(spans) == 1

    def test_noop_outside_collect_spans(self):
        build_fallback_response([make_hit()], retrieval_confidence=0.9, threshold=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/generation/test_fallback_response.py -k "records_generation_span or records_span_even or noop_outside" -v`
Expected: FAIL — `0 == 1`.

- [ ] **Step 3: Add the decorator**

In `src/generation/fallback_response.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.models import VectorStoreHit
from src.tracing.instrumentation import traced

FALLBACK_MESSAGE = (
    "I found some potentially related information, but not enough to "
    "answer confidently."
)


@dataclass(frozen=True)
class FallbackResponse:
    # ... unchanged
    message: str
    retrieved_summary: str
    documents_to_check: list[str]


def _document_label(hit: VectorStoreHit, *, ambiguous_titles: set[str]) -> str:
    if hit.title in ambiguous_titles:
        return f"{hit.title} ({hit.source_path})"
    return hit.title


@traced("generation")
def build_fallback_response(
    hits: list[VectorStoreHit],
    retrieval_confidence: float,
    threshold: float,
) -> FallbackResponse | None:
    """Return a `FallbackResponse` if `retrieval_confidence < threshold`, else `None`.
    ...
    """
    # ... body unchanged
```

(Only the import block and the `@traced("generation")` decorator line change
— `FallbackResponse`, `_document_label`, and the function body are unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/generation/test_fallback_response.py -v`
Expected: PASS (including `test_records_span_even_when_returning_none` — `traced()` records a span regardless of what the wrapped function returns, including `None`, since `default_serialize(None)` produces the valid JSON string `"null"`).

- [ ] **Step 5: Commit**

```bash
git add src/generation/fallback_response.py tests/unit/generation/test_fallback_response.py
git commit -m "feat(generation): instrument build_fallback_response with a generation span"
```

---

### Task 16: Full suite check + docs update

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/DECISIONS.md`

**Interfaces:** None — documentation only.

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: PASS — every test in the repo, old and new.

- [ ] **Step 2: Run lint and type checks on touched files**

```bash
ruff check src/tracing/ src/retrieval/ src/generation/ tests/unit/tracing/ tests/unit/retrieval/ tests/unit/generation/
ruff format --check src/tracing/context.py src/tracing/instrumentation.py src/tracing/models.py src/retrieval/dense_retriever.py src/retrieval/sparse_retriever.py src/retrieval/fusion.py src/retrieval/providers/reranker_sentence_transformers.py src/retrieval/providers/reranker_cohere.py src/retrieval/providers/reranker_voyage.py src/generation/citation_verifier.py src/generation/confidence_scorer.py src/generation/fallback_response.py src/generation/providers/citation_judge_anthropic.py src/generation/providers/citation_judge_openai.py src/generation/providers/completeness_judge_anthropic.py src/generation/providers/completeness_judge_openai.py
mypy src/tracing/ src/retrieval/ src/generation/
```

Expected: no errors on the files this plan touched. (Per repo convention,
ignore pre-existing `ruff format` failures in files this plan didn't touch —
don't reformat them.)

Fix anything that fails before proceeding.

- [ ] **Step 3: Update `docs/ARCHITECTURE.md`**

Add a new entry after the existing `## 2026-07-04 — Phase 3: Trace/Span Data Models (Complete)` section:

```markdown
## 2026-07-04 — Phase 3: Span Instrumentation (Complete)

### Wrapping Pipeline Steps in Spans

`src/tracing/context.py` and `src/tracing/instrumentation.py` give every
retrieval and generation function a way to record itself as a `Span`
without needing an orchestrator yet:

- **`collect_spans()`** (`context.py`) — a context manager holding a
  `ContextVar[list[Span] | None]`. Instrumented calls append their
  completed span to whichever list is active; outside any `collect_spans()`
  block, instrumentation is a no-op.
- **`span(step, input)`** (`instrumentation.py`) — a context manager for
  sites needing to attach detail mid-function (LLM prompt, token count)
  that a decorator can't infer from arguments/return value alone. Used
  directly inside each judge provider's `judge()` method.
- **`traced(step)`** (`instrumentation.py`) — a decorator built on `span()`
  that auto-serializes a function's bound arguments (`self` excluded) and
  return value. Applied as one line above each simpler pipeline function.

Applied at:

| Function | Step |
|---|---|
| `DenseRetriever.retrieve`, `SparseRetriever.retrieve`, `reciprocal_rank_fusion` | `retrieval` |
| `SentenceTransformersReranker.rerank`, `CohereReranker.rerank`, `VoyageReranker.rerank` | `ranking` |
| `verify_citations`, `AnthropicCitationJudge.judge`, `OpenAICitationJudge.judge` | `verification` |
| `score_confidence`, `AnthropicCompletenessJudge.judge`, `OpenAICompletenessJudge.judge`, `build_fallback_response` | `generation` |

`HybridRetriever.retrieve` itself is not separately wrapped — `Span` has no
parent/child relationship, so instrumenting both it and the four leaf calls
it makes would add a redundant, duplicate `retrieval`-step span.

`Span.confidence_score` is left unset by every function above: none of them
produce a discrete 1–5 rating today (retrieval similarity and
`ConfidenceScore.composite` are continuous 0–1 floats) — populating it is a
future orchestrator's decision.

**Public API:**

```python
from src.tracing.context import collect_spans

with collect_spans() as spans:
    hits = dense_retriever.retrieve("What is RRF?")
    reranked = reranker.rerank("What is RRF?", hits, top_n=5)
# spans now holds a Span for the retrieve() call and one for the rerank() call
```

Still no orchestrator exists to assemble these into a `Trace` per request,
or a JSON/SQLite writer to persist one — those remain future tasks.
```

- [ ] **Step 4: Update `docs/DECISIONS.md`**

Add a new entry after the existing `## 2026-07-04 — Trace/Span Data Models` section:

```markdown
## 2026-07-04 — Span Instrumentation

**`ContextVar`-based sink (`collect_spans()`), not a required `Trace` object per call** — No orchestrator exists yet to assemble one `Trace` per request (every generation-module function is still "a standalone, directly-callable unit" per the entries above). Requiring every instrumented call to build or receive a `Trace` would invent an API shape for an orchestrator that doesn't exist. `collect_spans()` instead yields a plain `list[Span]` that a future orchestrator wraps into a `Trace` itself; outside any `collect_spans()` block, instrumented calls run exactly as before, which is what keeps every pre-existing unit test (none of which set up tracing) passing unmodified.

**`span()` context manager for LLM call sites, `traced()` decorator for everything else** — A decorator can only see a function's arguments and return value, but the LLM prompt, raw response, and token usage a judge provider's `judge()` method produces exist only *inside* the method body, after the API call returns. `span()` is the primitive that supports attaching that mid-function detail (`s.llm_prompt = ...`, `s.token_count = ...`); `traced()` is sugar over `span()` for the common case where a function's arguments and return value are the whole story.

**Token counts extracted via `isinstance` checks, not `try`/`except`** — `unittest.mock.MagicMock` auto-vivifies any attribute access (`mock.usage.input_tokens` returns another `MagicMock`, not an `AttributeError`), so every existing judge-provider test that doesn't explicitly set `.usage` would otherwise leak a `MagicMock` into `Span.token_count` — which fails pydantic validation (`int | None`) and would break tests that never asked for tracing. `_extract_token_count` in each of the four judge provider files checks `isinstance(value, int)` and falls back to `None` rather than assuming the attribute access itself will fail.

**`HybridRetriever.retrieve` is not separately instrumented** — `Span` has no parent/child field, so `Trace.spans` is a flat list. Wrapping the coordinating method in addition to the four leaf calls it makes (`dense.retrieve`, `sparse.retrieve`, `reciprocal_rank_fusion`, `reranker.rerank`) would add a fifth, redundant `retrieval`-step span with no field to express "this one contains those."

**`Span.confidence_score` left unset by this task** — Populating it requires mapping the continuous 0–1 floats this codebase actually computes (retrieval similarity, `ConfidenceScore.composite`) onto the model's `1-5` int range. No such mapping is specified anywhere in the project docs; inventing one wasn't part of this task's scope, so it's left for whichever future orchestrator decides that conversion.
```

- [ ] **Step 5: Commit**

```bash
git add docs/ARCHITECTURE.md docs/DECISIONS.md
git commit -m "docs: document span instrumentation across project docs"
```

---

## Self-Review Notes

- **Spec coverage:** every in-scope item from `docs/superpowers/specs/2026-07-04-span-instrumentation-design.md` maps to a task — `Span.error` (Task 1), `collect_spans` (Task 2), `span()`/`default_serialize` (Task 3), `traced()` (Task 4), all nine `@traced` call sites (Tasks 5–11, 13, 15), all four judge providers' manual `span()` instrumentation (Tasks 12, 14), and doc updates (Task 16).
- **Type consistency:** `traced(step: PipelineStep)` and `span(step: PipelineStep, input: str)` signatures introduced in Tasks 3–4 are used identically (same parameter names/order) in every later task. `_SpanBuilder`'s field names (`output`, `llm_prompt`, `token_count`, `confidence_score`) match what Tasks 12 and 14 assign on the `s` object.
- **No placeholders:** every step shows the exact code to write and the exact command + expected result to run.
