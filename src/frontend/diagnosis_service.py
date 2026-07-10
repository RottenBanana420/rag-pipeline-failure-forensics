"""On-demand root-cause diagnosis for the trace view.

`run_diagnosis` makes one real LLM judge call per span in a trace's unhealthy
tail (`find_root_cause_span`), plus up to two more (`categorize_failure`,
`build_evidence_chain`) only if a root cause is actually found — real
Anthropic/OpenAI spend, per CLAUDE.md's "LLM Judge Cost Management" section.

This module holds no caching logic of its own: the caller (`app.py`) is
responsible for only invoking `run_diagnosis` on an explicit user action (the
"Diagnose root cause" button) and for caching the returned `DiagnosisResult`
in `st.session_state` keyed by `trace_id`, so repeat views of the same trace
within a session don't re-spend. This is the only module in `src/frontend/`
importing from `src/analysis/`, keeping the LLM-spend surface isolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.analysis.evidence_chain import (
    EvidenceChain,
    build_evidence_chain,
    make_evidence_chain_judge,
)
from src.analysis.failure_categorizer import (
    FailureCategoryVerdict,
    categorize_failure,
    make_failure_category_judge,
)
from src.analysis.root_cause import (
    RootCauseDiagnosis,
    find_root_cause_span,
    make_step_quality_judge,
)
from src.tracing.models import Trace

if TYPE_CHECKING:
    from src.config import Settings


@dataclass(frozen=True)
class DiagnosisResult:
    """The full outcome of an on-demand diagnosis run.

    `diagnosis is None` means no root cause was found (the trace's unhealthy
    tail never actually dipped below threshold) — in that case `category`
    and `evidence_chain` are also `None`, since there is nothing to
    categorize or narrate.
    """

    diagnosis: RootCauseDiagnosis | None
    category: FailureCategoryVerdict | None
    evidence_chain: EvidenceChain | None


def run_diagnosis(trace: Trace, settings: Settings) -> DiagnosisResult:
    """Run the full backward root-cause diagnosis pipeline for *trace*.

    Constructs a step-quality judge and calls `find_root_cause_span`. If no
    root cause is found, returns immediately without constructing the
    category/evidence-chain judges or calling them — avoiding two wasted LLM
    calls when there's nothing to categorize or narrate.
    """
    step_quality_judge = make_step_quality_judge(settings)
    diagnosis = find_root_cause_span(
        trace, step_quality_judge, threshold=settings.root_cause_quality_threshold
    )
    if diagnosis is None:
        return DiagnosisResult(diagnosis=None, category=None, evidence_chain=None)

    category_judge = make_failure_category_judge(settings)
    category = categorize_failure(diagnosis, category_judge)

    evidence_chain_judge = make_evidence_chain_judge(settings)
    evidence_chain = build_evidence_chain(diagnosis, category, evidence_chain_judge)

    return DiagnosisResult(
        diagnosis=diagnosis, category=category, evidence_chain=evidence_chain
    )
