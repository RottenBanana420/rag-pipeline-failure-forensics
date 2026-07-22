#!/usr/bin/env python3
"""Execute the full evaluation suite and print metrics.

Builds the eval-isolated retriever/generator/judges once, runs every (or a
filtered subset of) golden test case through `src.evaluation.runner`, prints
a summary table, saves the report, and diffs it against the previous run to
flag regressions. Spends real LLM API money — see CLAUDE.md's "LLM Judge
Cost Management" section for cadence guidance.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.config import Settings
from src.evaluation.answer_correctness import make_answer_correctness_judge
from src.evaluation.corpus import ensure_golden_corpus_indexed
from src.evaluation.dataset import VALID_CATEGORIES, filter_cases, load_golden_dataset
from src.evaluation.faithfulness import make_faithfulness_judge
from src.evaluation.models import EvalReport, MetricAggregate
from src.evaluation.regression_tracker import (
    compare_reports,
    generate_run_id,
    load_latest_report,
    save_report,
)
from src.evaluation.runner import aggregate, run_eval
from src.frontend.query_service import build_hybrid_retriever
from src.generation.answer_generator import make_answer_generator
from src.generation.citation_verifier import make_citation_judge


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the golden-dataset eval suite.")
    parser.add_argument("--category", choices=sorted(VALID_CATEGORIES), default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _print_summary(
    overall: MetricAggregate, by_category: dict[str, MetricAggregate]
) -> None:
    header = (
        f"{'Scope':<12} {'Correctness':>12} {'Faithfulness':>13} "
        f"{'Relevance':>10} {'Citations':>10}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'overall':<12} {_fmt(overall.answer_correctness):>12} "
        f"{_fmt(overall.faithfulness):>13} {_fmt(overall.retrieval_relevance):>10} "
        f"{_fmt(overall.citation_accuracy):>10}"
    )
    for category in sorted(by_category):
        agg = by_category[category]
        print(
            f"{category:<12} {_fmt(agg.answer_correctness):>12} "
            f"{_fmt(agg.faithfulness):>13} {_fmt(agg.retrieval_relevance):>10} "
            f"{_fmt(agg.citation_accuracy):>10}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    settings = Settings()
    if args.output_dir is not None:
        settings = settings.model_copy(update={"eval_output_dir": args.output_dir})

    cases = filter_cases(
        load_golden_dataset(), category=args.category, limit=args.limit
    )
    if not cases:
        print("No golden cases matched the given filters.", file=sys.stderr)
        return 1

    eval_settings = settings.model_copy(
        update={"chroma_persist_dir": settings.eval_chroma_persist_dir}
    )
    ensure_golden_corpus_indexed(eval_settings)
    retriever = build_hybrid_retriever(eval_settings).hybrid

    generator = make_answer_generator(settings)
    citation_judge = make_citation_judge(settings)
    faithfulness_judge = make_faithfulness_judge(settings)
    correctness_judge = make_answer_correctness_judge(settings)

    results = run_eval(
        cases,
        retriever,
        generator,
        citation_judge,
        faithfulness_judge,
        correctness_judge,
    )
    overall, by_category = aggregate(results)
    _print_summary(overall, by_category)

    report = EvalReport(
        run_id=generate_run_id(),
        timestamp=datetime.now(UTC),
        results=results,
        overall=overall,
        by_category=by_category,
    )
    saved_path = save_report(report, settings)
    previous = load_latest_report(settings, exclude=saved_path)
    findings = compare_reports(previous, report, settings.eval_regression_threshold)

    error_count = sum(1 for r in results if r.error is not None)
    if error_count:
        print(f"\n{error_count} case(s) errored during this run.", file=sys.stderr)

    if findings:
        print("\nRegressions detected:")
        for finding in findings:
            print(
                f"  [{finding.scope}] {finding.metric}: "
                f"{finding.previous_value:.2f} -> {finding.current_value:.2f} "
                f"(drop {finding.drop:.2f})"
            )
    else:
        print("\nNo regressions detected.")

    return 1 if (findings or error_count) else 0


if __name__ == "__main__":
    sys.exit(main())
