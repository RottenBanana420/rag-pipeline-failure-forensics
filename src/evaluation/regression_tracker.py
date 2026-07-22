"""Persist EvalReport runs and detect metric regressions across runs.

One JSON file per run under `settings.eval_output_dir`, named by a
lexicographically-sortable UTC-timestamp run id — no separate index file
needed, `sorted(glob("*.json"))[-1]` finds the latest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.evaluation.models import EvalReport, MetricAggregate

if TYPE_CHECKING:
    from src.config import Settings

_METRIC_NAMES = (
    "answer_correctness",
    "faithfulness",
    "retrieval_relevance",
    "citation_accuracy",
)


def generate_run_id() -> str:
    """Lexicographically-sortable UTC timestamp run id, e.g. ``20260715T120000Z``."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def save_report(report: EvalReport, settings: Settings) -> Path:
    """Write *report* to `{settings.eval_output_dir}/{report.run_id}.json`."""
    output_dir = Path(settings.eval_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.run_id}.json"
    path.write_text(report.model_dump_json(indent=2))
    return path


def load_latest_report(
    settings: Settings, exclude: Path | None = None
) -> EvalReport | None:
    """Load the most recently saved report, or `None` if there isn't one.

    *exclude* skips a specific file (e.g. the one just written this run),
    so a fresh run can diff against the prior run rather than itself.
    """
    output_dir = Path(settings.eval_output_dir)
    if not output_dir.exists():
        return None

    files = sorted(output_dir.glob("*.json"))
    if exclude is not None:
        files = [f for f in files if f.resolve() != exclude.resolve()]
    if not files:
        return None

    return EvalReport.model_validate_json(files[-1].read_text())


@dataclass(frozen=True)
class RegressionFinding:
    scope: str  # "overall" or a category name
    metric: str
    previous_value: float
    current_value: float
    drop: float


def _compare_aggregates(
    scope: str, previous: MetricAggregate, current: MetricAggregate, threshold: float
) -> list[RegressionFinding]:
    findings: list[RegressionFinding] = []
    for metric in _METRIC_NAMES:
        prev_value = getattr(previous, metric)
        curr_value = getattr(current, metric)
        if prev_value is None or curr_value is None:
            continue
        drop = prev_value - curr_value
        if drop > threshold:
            findings.append(
                RegressionFinding(
                    scope=scope,
                    metric=metric,
                    previous_value=prev_value,
                    current_value=curr_value,
                    drop=drop,
                )
            )
    return findings


def compare_reports(
    previous: EvalReport | None, current: EvalReport, threshold: float
) -> list[RegressionFinding]:
    """Flag any metric (overall or per-category) that dropped by more than *threshold*.

    Returns `[]` if there's no previous report (first-ever run). A category
    present only in `current` (or only in `previous`) has nothing to compare
    against and is skipped, not treated as a regression.
    """
    if previous is None:
        return []

    findings = _compare_aggregates(
        "overall", previous.overall, current.overall, threshold
    )
    for category, curr_agg in current.by_category.items():
        prev_agg = previous.by_category.get(category)
        if prev_agg is None:
            continue
        findings.extend(_compare_aggregates(category, prev_agg, curr_agg, threshold))
    return findings
