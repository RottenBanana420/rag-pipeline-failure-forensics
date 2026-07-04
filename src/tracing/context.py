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
