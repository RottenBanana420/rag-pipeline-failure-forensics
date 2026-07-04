from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

PipelineStep = Literal[
    "ingestion", "retrieval", "ranking", "generation", "verification"
]


class Span(BaseModel):
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    step: PipelineStep
    input: str
    output: str
    llm_prompt: str | None = None
    token_count: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0.0)
    confidence_score: int | None = Field(default=None, ge=1, le=5)
    error: str | None = None


TraceStatus = Literal["success", "failure", "degraded"]


class Trace(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    spans: list[Span] = Field(default_factory=list)
    final_output: str | None = None
    status: TraceStatus
