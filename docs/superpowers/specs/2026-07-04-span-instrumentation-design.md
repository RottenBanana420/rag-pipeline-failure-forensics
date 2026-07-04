# Span Instrumentation Design

## Context

`src/tracing/models.py` already defines `Trace`/`Span` as plain, standalone
pydantic models (see `docs/superpowers/specs/2026-07-04-trace-span-models-design.md`).
Nothing yet constructs a `Span` automatically as pipeline code runs. This spec
covers that next step: a context manager + decorator that instrument existing
retrieval and generation functions, per `CLAUDE.md`'s tracing module layout
("context manager, decorator ... — planned").

No orchestrator exists yet to run a full request end-to-end and assemble one
`Trace` per request (every generation-module function is still described in
`CLAUDE.md`/`docs/DECISIONS.md` as "a standalone, directly-callable unit").
This spec does not build that orchestrator. It builds the instrumentation
primitives and applies them at each existing pipeline function, so that
whenever the orchestrator does exist, wiring it into tracing is immediate.

## Scope

In scope:
- `Span` model gains an `error: str | None = None` field.
- `src/tracing/instrumentation.py`: `span()` context manager, `traced()`
  decorator, `default_serialize()` helper.
- `src/tracing/context.py`: `collect_spans()`, the contextvar-based sink that
  `span()`/`traced()` append completed spans into.
- Applying `@traced` to: `DenseRetriever.retrieve`, `SparseRetriever.retrieve`,
  `reciprocal_rank_fusion`, each reranker provider's `rerank()`
  (`sentence_transformers`/`cohere`/`voyage`), `verify_citations`,
  `score_confidence`, `build_fallback_response`.
- Applying `span()` manually inside each judge provider's `judge()` method
  (`citation_judge_anthropic`/`citation_judge_openai`/
  `completeness_judge_anthropic`/`completeness_judge_openai`) to capture the
  real LLM prompt, raw response, and token usage.
- Unit tests for the new instrumentation module and updated tests for every
  touched call site confirming a span is recorded (and, for judge providers,
  that prompt/token data is captured).

Out of scope (future tasks):
- A `Trace`-per-request orchestrator that calls `collect_spans()` around a
  full pipeline run and decides `Trace.status`/`final_output`.
- JSON file writer / SQLite indexer for completed traces.
- Backward root-cause analysis walker.
- Populating `Span.confidence_score` (1–5) — no function in scope here
  produces a discrete 1–5 rating today (retrieval similarity and
  `ConfidenceScore.composite` are continuous 0–1 floats); mapping one to the
  other is a future orchestrator's decision, not this task's.
- Wrapping `HybridRetriever.retrieve` itself — see Design.

## Design

### `Span.error`

```python
class Span(BaseModel):
    ...
    error: str | None = None
```

Added so a span can record "this step raised" without overloading `output`
(which stays a clean record of a successful return value). `None` when the
wrapped call succeeded.

### `src/tracing/context.py` — span collection sink

```python
_current_spans: ContextVar[list[Span] | None] = ContextVar("_current_spans", default=None)

@contextmanager
def collect_spans() -> Iterator[list[Span]]:
    spans: list[Span] = []
    token = _current_spans.set(spans)
    try:
        yield spans
    finally:
        _current_spans.reset(token)

def _active_sink() -> list[Span] | None:
    return _current_spans.get()
```

A future orchestrator uses this as:
```python
with collect_spans() as spans:
    hits = hybrid_retriever.retrieve(query)
    ...
trace = Trace(spans=spans, final_output=answer, status="success")
```
Outside any `collect_spans()` block, `_active_sink()` returns `None` and
instrumented functions run exactly as before — a completed `Span` is built
but has nowhere to go, so it's simply dropped. This is what keeps every
existing unit test (which calls e.g. `dense_retriever.retrieve()` with no
tracing setup at all) passing unmodified.

### `src/tracing/instrumentation.py` — `span()` and `traced()`

`span()` is the core primitive — a context manager for sites that need to
attach detail (LLM prompt, raw response, token count) mid-function, which a
decorator can't infer from arguments/return value alone:

```python
@dataclass
class _SpanBuilder:
    step: PipelineStep
    input: str
    output: str = ""
    llm_prompt: str | None = None
    token_count: int | None = None
    confidence_score: int | None = None

@contextmanager
def span(step: PipelineStep, input: str) -> Iterator[_SpanBuilder]:
    builder = _SpanBuilder(step=step, input=input)
    start = time.perf_counter()
    error: str | None = None
    try:
        yield builder
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        completed = Span(
            step=builder.step,
            input=builder.input,
            output=builder.output,
            llm_prompt=builder.llm_prompt,
            token_count=builder.token_count,
            latency_ms=latency_ms,
            confidence_score=builder.confidence_score,
            error=error,
        )
        sink = _active_sink()
        if sink is not None:
            sink.append(completed)
```

Usage inside a judge provider:
```python
def judge(self, claim: str, evidence: str) -> JudgeVerdict:
    prompt = build_judge_prompt(claim, evidence)
    with span("verification", input=f"claim={claim!r}\nevidence={evidence!r}") as s:
        s.llm_prompt = f"{CITATION_JUDGE_SYSTEM_PROMPT}\n\n{prompt.user}"
        response = self._client.messages.parse(...)
        s.token_count = _extract_token_count(response)
        parsed = response.parsed_output
        if parsed is None:
            raise RuntimeError(...)
        s.output = default_serialize(parsed)
        return parsed
```
(`_extract_token_count` is provider-specific — Anthropic's and OpenAI's SDKs
expose usage differently; exact field names confirmed via Context7 before
writing each provider's version, not assumed.)

`traced()` is sugar over `span()` for functions where input/output are
just the arguments and return value:

```python
def traced(step: PipelineStep):
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

Applying it is one line:
```python
@traced("retrieval")
def retrieve(self, query: str, k: int = _DEFAULT_K) -> list[VectorStoreHit]:
    ...
```

### `default_serialize`

```python
def default_serialize(value: object) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json.dumps(dataclasses.asdict(value), default=str)
    if isinstance(value, list):
        return json.dumps([_serialize_item(v) for v in value], default=str)
    if isinstance(value, dict):
        return json.dumps({k: _serialize_item(v) for k, v in value.items()}, default=str)
    return repr(value)
```
(`_serialize_item` mirrors the same dispatch for list/dict elements —
dataclass or pydantic model to a JSON-able dict, else the value as-is /
`repr()`.) Best-effort, not a strict schema — good enough for a human
reading a trace, not meant to be round-tripped back into Python objects.

### Why `HybridRetriever.retrieve` isn't separately wrapped

`Span` has no parent/child relationship — `Trace.spans` is a flat list.
`HybridRetriever.retrieve` calls `dense.retrieve` → `sparse.retrieve` →
`reciprocal_rank_fusion` → `reranker.rerank`, each already instrumented.
Wrapping the coordinating method too would add a fifth, redundant
`"retrieval"`-step span whose input/output duplicate information already
visible across its four children, with no field to express "this one
contains those." The four leaf calls are the real steps from the
architecture doc's pipeline diagram.

## Testing

- `tests/unit/tracing/test_instrumentation.py`: `span()` records latency,
  captures `error` and re-raises on exception, is a no-op when no
  `collect_spans()` is active, appends to the active sink when one is;
  `traced()` auto-serializes args (excluding `self`) and return value,
  preserves the wrapped function's return value and propagates exceptions.
- `tests/unit/tracing/test_context.py`: `collect_spans()` yields an
  appendable list, resets the contextvar on exit (including on exception),
  nested `collect_spans()` calls don't leak into each other.
- Existing test files for each touched call site (`test_dense_retriever.py`,
  `test_sparse_retriever.py`, `test_fusion.py`, `test_reranker_*.py`,
  `test_citation_verifier.py`, `test_confidence_scorer.py`,
  `test_fallback_response.py`, `test_citation_judge_*.py`,
  `test_completeness_judge_*.py`) get one additional test each: wrap the call
  in `collect_spans()`, assert exactly one span is recorded with the
  expected `step`, and — for the four judge providers — assert `llm_prompt`
  is non-empty and `token_count` is set from the mocked response's usage
  data.
