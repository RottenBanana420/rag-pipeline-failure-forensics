# Trace/Span Models Design

## Context

Phase 3 (tracing) needs a data model that becomes the complete record of what
happened during a RAG request. Every request gets a unique `trace_id`, and the
request's pipeline steps (ingestion, retrieval, ranking, generation,
verification) each produce a `Span`.

This spec covers **only the `Trace` and `Span` data models**
(`src/tracing/models.py`). It does not cover the context manager, decorator,
or JSON/SQLite writers described elsewhere in `CLAUDE.md`'s tracing module
layout — those are separate, later tasks that will consume these models.

## Scope

In scope:
- `Span` pydantic model
- `Trace` pydantic model
- Supporting `Literal` type aliases (`PipelineStep`, `TraceStatus`)
- Unit tests for both models

Out of scope (future tasks):
- Context manager / decorator that times a pipeline step and constructs a `Span`
- JSON file writer / SQLite indexer for completed traces
- Backward root-cause analysis walker
- Any orchestrator that wires spans into the actual pipeline functions

## Design

### `PipelineStep` and `TraceStatus`

```python
PipelineStep = Literal["ingestion", "retrieval", "ranking", "generation", "verification"]
TraceStatus = Literal["success", "failure", "degraded"]
```

Closed `Literal`s, matching the existing convention for fixed enumerations in
this codebase (`ChunkingStrategy` in `src/ingestion/models.py`,
`source_format` on `ProcessedDocument`). The five steps and three statuses are
fixed by the architecture doc; typos should fail validation rather than
silently producing an uncategorizable trace.

### `Span`

```python
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

- `input`/`output` are `str`. The architecture doc specifies spans record
  "serialized input/output" — the actual serialization of arbitrary pipeline
  function arguments/return values is the future decorator's job, not this
  model's. This model just stores the already-serialized string.
- `confidence_score` is optional (`1`-`5` when present) because not every
  step naturally produces one (e.g. an ingestion span has no LLM-judged
  score, but a generation or verification span might).
- `token_count` is optional since only LLM-calling steps consume tokens.
- `span_id` auto-generates a UUID4 hex string so callers don't need to
  supply one.

### `Trace`

```python
class Trace(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    spans: list[Span] = Field(default_factory=list)
    final_output: str | None = None
    status: TraceStatus
```

- `trace_id` auto-generates a UUID4 hex string, mirroring `Span.span_id`.
- `spans` defaults to an empty list — a future context manager will append
  `Span`s as pipeline steps complete.
- `final_output` defaults to `None` (e.g. a trace for a failed request may
  never produce output).
- `status` is required, with no default — the object described here
  represents "the complete record of what happened," so callers must state
  the outcome explicitly rather than relying on an implicit/optimistic
  default.

Both models are plain (non-frozen) pydantic `BaseModel`s, matching the
existing convention for persisted/serializable domain objects
(`ProcessedDocument`, `Chunk` in `src/ingestion/models.py`) rather than the
frozen-dataclass convention used for immutable in-memory result values
(`ConfidenceScore`, `FallbackResponse` in `src/generation/`). Pydantic gives
free JSON serialization (`model_dump_json()`), which the future JSON/SQLite
writers will need.

## Testing

`tests/unit/tracing/test_models.py`:
- `Span` constructs with only required fields (`step`, `input`, `output`,
  `latency_ms`); optional fields default correctly; `span_id` auto-generates
  and is unique across instances.
- `Span` rejects `confidence_score` outside `1..5`, negative `token_count`,
  negative `latency_ms`, and an invalid `step` literal.
- `Trace` constructs with only `status` required; `trace_id` auto-generates
  and is unique across instances; `spans` defaults to an empty list.
- `Trace` holding a list of `Span`s round-trips through
  `model_dump()`/`model_validate()` (and `model_dump_json()`/
  `model_validate_json()`) without data loss — this is the shape the future
  JSON writer will depend on.
- `Trace` rejects an invalid `status` literal.
