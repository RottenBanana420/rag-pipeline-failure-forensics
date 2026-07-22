"""Unit tests for src/evaluation/regression_tracker.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.evaluation.models import EvalReport, MetricAggregate


def make_report(run_id: str, overall: MetricAggregate, by_category=None) -> EvalReport:
    return EvalReport(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        overall=overall,
        by_category=by_category or {},
    )


@pytest.fixture
def eval_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("EVAL_OUTPUT_DIR", str(tmp_path / "eval-runs"))
    from src.config import Settings

    return Settings()


class TestSaveReport:
    def test_writes_json_file_named_by_run_id(self, eval_settings):
        from src.evaluation.regression_tracker import save_report

        report = make_report(
            "20260715T120000Z", MetricAggregate(answer_correctness=1.0)
        )

        path = save_report(report, eval_settings)

        assert path.name == "20260715T120000Z.json"
        assert path.exists()

    def test_creates_output_dir_if_missing(self, eval_settings):
        from src.evaluation.regression_tracker import save_report

        assert not eval_settings.eval_output_dir.exists()

        report = make_report("run-1", MetricAggregate())
        save_report(report, eval_settings)

        assert eval_settings.eval_output_dir.exists()

    def test_saved_file_round_trips_via_model_validate_json(self, eval_settings):
        from src.evaluation.regression_tracker import save_report

        report = make_report("run-1", MetricAggregate(answer_correctness=0.75))

        path = save_report(report, eval_settings)
        loaded = EvalReport.model_validate_json(path.read_text())

        assert loaded == report


class TestLoadLatestReport:
    def test_returns_none_when_no_reports_saved(self, eval_settings):
        from src.evaluation.regression_tracker import load_latest_report

        assert load_latest_report(eval_settings) is None

    def test_returns_none_when_output_dir_does_not_exist(self, eval_settings):
        from src.evaluation.regression_tracker import load_latest_report

        assert load_latest_report(eval_settings) is None

    def test_returns_most_recent_by_run_id_sort_order(self, eval_settings):
        from src.evaluation.regression_tracker import load_latest_report, save_report

        save_report(
            make_report("20260715T100000Z", MetricAggregate(answer_correctness=0.5)),
            eval_settings,
        )
        save_report(
            make_report("20260715T120000Z", MetricAggregate(answer_correctness=0.9)),
            eval_settings,
        )

        latest = load_latest_report(eval_settings)

        assert latest.run_id == "20260715T120000Z"
        assert latest.overall.answer_correctness == 0.9

    def test_exclude_path_skips_that_file(self, eval_settings):
        from src.evaluation.regression_tracker import load_latest_report, save_report

        save_report(
            make_report("20260715T100000Z", MetricAggregate(answer_correctness=0.5)),
            eval_settings,
        )
        just_saved = save_report(
            make_report("20260715T120000Z", MetricAggregate(answer_correctness=0.9)),
            eval_settings,
        )

        latest = load_latest_report(eval_settings, exclude=just_saved)

        assert latest.run_id == "20260715T100000Z"


class TestCompareReports:
    def test_previous_none_returns_empty(self):
        from src.evaluation.regression_tracker import compare_reports

        current = make_report("run-2", MetricAggregate(answer_correctness=0.5))

        assert compare_reports(None, current, threshold=0.05) == []

    def test_no_regression_returns_empty(self):
        from src.evaluation.regression_tracker import compare_reports

        previous = make_report("run-1", MetricAggregate(answer_correctness=0.8))
        current = make_report("run-2", MetricAggregate(answer_correctness=0.85))

        assert compare_reports(previous, current, threshold=0.05) == []

    def test_detects_overall_metric_drop_above_threshold(self):
        from src.evaluation.regression_tracker import compare_reports

        previous = make_report("run-1", MetricAggregate(answer_correctness=0.9))
        current = make_report("run-2", MetricAggregate(answer_correctness=0.7))

        findings = compare_reports(previous, current, threshold=0.05)

        assert len(findings) == 1
        assert findings[0].scope == "overall"
        assert findings[0].metric == "answer_correctness"
        assert findings[0].previous_value == pytest.approx(0.9)
        assert findings[0].current_value == pytest.approx(0.7)
        assert findings[0].drop == pytest.approx(0.2)

    def test_ignores_drop_at_or_below_threshold(self):
        from src.evaluation.regression_tracker import compare_reports

        previous = make_report("run-1", MetricAggregate(answer_correctness=0.90))
        current = make_report("run-2", MetricAggregate(answer_correctness=0.86))

        assert compare_reports(previous, current, threshold=0.05) == []

    def test_detects_per_category_drop(self):
        from src.evaluation.regression_tracker import compare_reports

        previous = make_report(
            "run-1",
            MetricAggregate(),
            by_category={"lookup": MetricAggregate(faithfulness=0.9)},
        )
        current = make_report(
            "run-2",
            MetricAggregate(),
            by_category={"lookup": MetricAggregate(faithfulness=0.6)},
        )

        findings = compare_reports(previous, current, threshold=0.05)

        assert len(findings) == 1
        assert findings[0].scope == "lookup"
        assert findings[0].metric == "faithfulness"

    def test_skips_metrics_with_none_values_on_either_side(self):
        from src.evaluation.regression_tracker import compare_reports

        previous = make_report("run-1", MetricAggregate(retrieval_relevance=None))
        current = make_report("run-2", MetricAggregate(retrieval_relevance=0.5))

        assert compare_reports(previous, current, threshold=0.05) == []

    def test_category_only_in_current_is_not_compared(self):
        from src.evaluation.regression_tracker import compare_reports

        previous = make_report("run-1", MetricAggregate())
        current = make_report(
            "run-2",
            MetricAggregate(),
            by_category={"edge_case": MetricAggregate(answer_correctness=0.1)},
        )

        assert compare_reports(previous, current, threshold=0.05) == []


class TestRegressionFindingIsFrozenDataclass:
    def test_frozen(self):
        import dataclasses

        from src.evaluation.regression_tracker import RegressionFinding

        finding = RegressionFinding(
            scope="overall",
            metric="answer_correctness",
            previous_value=0.9,
            current_value=0.7,
            drop=0.2,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            finding.drop = 0.5  # type: ignore[misc]
