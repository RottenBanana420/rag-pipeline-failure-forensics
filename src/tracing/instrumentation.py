from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

from pydantic import BaseModel

from src.tracing.context import _active_sink
from src.tracing.models import PipelineStep, Span

P = ParamSpec("P")
T = TypeVar("T")


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


def traced(step: PipelineStep) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator: wrap a function/method so each call records a `Span`.

    Auto-serializes the call's bound arguments (`self` excluded, defaults
    applied) as `input` via `default_serialize`, and the return value as
    `output`. For sites needing to attach LLM prompt/response/token detail
    that isn't derivable from arguments/return value alone, use `span()`
    directly instead.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
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
