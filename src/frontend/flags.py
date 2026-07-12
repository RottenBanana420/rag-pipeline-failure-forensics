"""Persistence for human "flag as bad output" reviews — confirm or override a
root-cause diagnosis, creating the feedback signal PROJECT_SPEC.md's Phase 5
"flagging interface" item calls for.

One JSON file per trace (`{flags_dir}/{trace_id}.json`), mirroring
`src.frontend.corrections`'s one-file-per-trace-id convention. Unlike
corrections (nested per span_id, since a trace can have many span-level
corrections), a flag is trace-level: the file holds a single `FlagRecord`,
not a mapping.

`DiagnosisSummary` is a flattened, JSON-serializable snapshot of a
`DiagnosisResult` (`src.frontend.diagnosis_service`) — plain str/int fields
only, no live `Span`/pydantic objects, the same rationale `EvidenceEntry`
(`src.analysis.evidence_chain`) already uses to keep a downstream consumer
decoupled from `root_cause.py`'s own dataclasses. `HumanReview` always
carries a full span_id/category/note triple, even when the human clicks
"Confirm diagnosis": the algorithm's own span/category are copied across
verbatim rather than modeling confirmed/overridden as separate shapes, so a
downstream reader never has to branch on `confirmed` to find the human's
actual verdict.

Serialized via `dataclasses.asdict` + manual reconstruction, not a new
dependency (no `dacite`/`cattrs`/`dataclasses-json` is declared in
pyproject.toml) — the same hand-rolled-JSON-shape approach `corrections.py`
already uses for its own (simpler) dict shape.

Field names are chosen so a future Phase 6 orchestrator (PROJECT_SPEC.md
item 4, "auto-generate eval cases from production flags") can build an eval
test case directly from a `FlagRecord`: the original question isn't stored
here (it lives in the `Trace` JSON itself, under the same trace_id), but
`human_review.category` and `human_review.span_id` (-> that span's
`Span.step`, the failing step) map directly onto that item's "failure
category" and "the step where it failed." Building that orchestrator is out
of scope here — this module only persists the record.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.analysis.failure_categorizer import FailureCategory
from src.tracing.models import PipelineStep

if TYPE_CHECKING:
    from src.frontend.diagnosis_service import DiagnosisResult


@dataclass(frozen=True)
class DiagnosisSummary:
    """Flattened, JSON-serializable snapshot of a DiagnosisResult.

    Only ever built when `DiagnosisResult.diagnosis is not None` — in that
    case `run_diagnosis` guarantees `category`/`evidence_chain` are also
    populated (see `diagnosis_service.py`), so every field here is a plain,
    required str/int, not Optional.
    """

    root_cause_span_id: str
    step: PipelineStep
    score: int
    rationale: str
    category: FailureCategory
    category_rationale: str
    narrative: str


@dataclass(frozen=True)
class HumanReview:
    """The human's finalized verdict — always a full re-label.

    `confirmed=True` when the human accepted the algorithm's own diagnosis
    verbatim ("Confirm diagnosis"); `False` when they picked a different
    span/category/note via the override form. `span_id`/`category`/`note`
    are populated in both cases.
    """

    confirmed: bool
    span_id: str
    category: FailureCategory
    note: str


@dataclass(frozen=True)
class FlagRecord:
    """A trace's persisted "bad output" flag: when it happened, what the
    algorithm diagnosed (if anything — `None` when the diagnosis run found
    no unhealthy span, e.g. an override on an already-healthy trace), and
    the human's final verdict."""

    flagged_at: str
    diagnosis: DiagnosisSummary | None
    human_review: HumanReview


def diagnosis_summary_from_result(result: DiagnosisResult) -> DiagnosisSummary | None:
    """Flatten a DiagnosisResult into a DiagnosisSummary.

    Returns None when `result.diagnosis is None` (no root cause found) —
    mirrors `root_cause_span_id_from_diagnosis`'s Optional-in-Optional-out
    convention. `result.category`/`result.evidence_chain` are guaranteed
    non-None whenever `result.diagnosis` is non-None (`run_diagnosis`'s own
    invariant), asserted here so both mypy and a misuse at runtime are caught.
    """
    if result.diagnosis is None:
        return None
    assert result.category is not None
    assert result.evidence_chain is not None
    span = result.diagnosis.root_cause_span
    return DiagnosisSummary(
        root_cause_span_id=span.span_id,
        step=span.step,
        score=result.diagnosis.score,
        rationale=result.diagnosis.rationale,
        category=result.category.category,
        category_rationale=result.category.rationale,
        narrative=result.evidence_chain.narrative,
    )


def _flag_path(trace_id: str, flags_dir: Path) -> Path:
    return flags_dir / f"{trace_id}.json"


def save_flag(trace_id: str, record: FlagRecord, flags_dir: Path) -> None:
    """Persist *record* as trace_id's flag, overwriting any prior flag."""
    flags_dir.mkdir(parents=True, exist_ok=True)
    path = _flag_path(trace_id, flags_dir)
    path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")


def load_flag(trace_id: str, flags_dir: Path) -> FlagRecord | None:
    """Return the persisted flag for trace_id, or None if never flagged."""
    path = _flag_path(trace_id, flags_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    diagnosis = (
        DiagnosisSummary(**data["diagnosis"]) if data["diagnosis"] is not None else None
    )
    return FlagRecord(
        flagged_at=data["flagged_at"],
        diagnosis=diagnosis,
        human_review=HumanReview(**data["human_review"]),
    )
