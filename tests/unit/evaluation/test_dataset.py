"""Unit tests for src/evaluation/dataset.py (load_golden_dataset / filter_cases)."""

from __future__ import annotations

import json


def _write_dataset(tmp_path, entries: list[dict]):
    path = tmp_path / "qa_dataset.json"
    path.write_text(json.dumps(entries))
    return path


SYNTHETIC_ENTRIES = [
    {
        "id": "qa-001",
        "question": "Who founded Northwind?",
        "expected_answer": "Jane Doe founded Northwind in 2015.",
        "category": "lookup",
        "source_documents": ["01-onboarding-guide.md"],
        "source_sections": ["Welcome & Team Structure"],
        "notes": None,
    },
    {
        "id": "qa-021",
        "question": "How does local dev relate to service ownership?",
        "expected_answer": "Combined answer across two docs.",
        "category": "multi_hop",
        "source_documents": ["01-onboarding-guide.md", "02-architecture-overview.md"],
        "source_sections": ["Local Development Setup", "Service Ownership Table"],
        "notes": None,
    },
    {
        "id": "qa-032",
        "question": "What is the refund policy?",
        "expected_answer": "The corpus does not cover this.",
        "category": "no_answer",
        "source_documents": [],
        "source_sections": [],
        "notes": "Deliberately uncovered.",
    },
]


class TestLoadGoldenDataset:
    def test_loads_entries_from_given_path(self, tmp_path):
        from src.evaluation.dataset import load_golden_dataset

        path = _write_dataset(tmp_path, SYNTHETIC_ENTRIES)

        cases = load_golden_dataset(path)

        assert len(cases) == 3

    def test_parses_lookup_case_fields(self, tmp_path):
        from src.evaluation.dataset import load_golden_dataset

        path = _write_dataset(tmp_path, SYNTHETIC_ENTRIES)

        cases = load_golden_dataset(path)
        lookup = next(c for c in cases if c.id == "qa-001")

        assert lookup.question == "Who founded Northwind?"
        assert lookup.expected_answer == "Jane Doe founded Northwind in 2015."
        assert lookup.category == "lookup"
        assert lookup.source_documents == ["01-onboarding-guide.md"]
        assert lookup.source_sections == ["Welcome & Team Structure"]
        assert lookup.notes is None

    def test_parses_multi_hop_case_with_parallel_arrays(self, tmp_path):
        from src.evaluation.dataset import load_golden_dataset

        path = _write_dataset(tmp_path, SYNTHETIC_ENTRIES)

        cases = load_golden_dataset(path)
        multi_hop = next(c for c in cases if c.id == "qa-021")

        assert len(multi_hop.source_documents) == 2
        assert len(multi_hop.source_sections) == 2
        assert multi_hop.source_documents[1] == "02-architecture-overview.md"
        assert multi_hop.source_sections[1] == "Service Ownership Table"

    def test_parses_no_answer_case_with_empty_lists(self, tmp_path):
        from src.evaluation.dataset import load_golden_dataset

        path = _write_dataset(tmp_path, SYNTHETIC_ENTRIES)

        cases = load_golden_dataset(path)
        no_answer = next(c for c in cases if c.id == "qa-032")

        assert no_answer.source_documents == []
        assert no_answer.source_sections == []
        assert no_answer.notes == "Deliberately uncovered."

    def test_default_path_loads_real_golden_dataset(self):
        from src.evaluation.dataset import load_golden_dataset

        cases = load_golden_dataset()

        assert len(cases) >= 50
        assert all(c.id.startswith("qa-") for c in cases)


class TestFilterCases:
    def test_no_filters_returns_all(self, tmp_path):
        from src.evaluation.dataset import filter_cases, load_golden_dataset

        cases = load_golden_dataset(_write_dataset(tmp_path, SYNTHETIC_ENTRIES))

        assert filter_cases(cases) == cases

    def test_category_filter_returns_only_matching(self, tmp_path):
        from src.evaluation.dataset import filter_cases, load_golden_dataset

        cases = load_golden_dataset(_write_dataset(tmp_path, SYNTHETIC_ENTRIES))

        filtered = filter_cases(cases, category="multi_hop")

        assert len(filtered) == 1
        assert filtered[0].id == "qa-021"

    def test_limit_truncates_to_first_n(self, tmp_path):
        from src.evaluation.dataset import filter_cases, load_golden_dataset

        cases = load_golden_dataset(_write_dataset(tmp_path, SYNTHETIC_ENTRIES))

        filtered = filter_cases(cases, limit=2)

        assert len(filtered) == 2
        assert filtered == cases[:2]

    def test_category_and_limit_combined(self, tmp_path):
        from src.evaluation.dataset import filter_cases, load_golden_dataset

        entries = SYNTHETIC_ENTRIES + [
            {**SYNTHETIC_ENTRIES[0], "id": "qa-002"},
        ]
        cases = load_golden_dataset(_write_dataset(tmp_path, entries))

        filtered = filter_cases(cases, category="lookup", limit=1)

        assert len(filtered) == 1
        assert filtered[0].id == "qa-001"

    def test_unmatched_category_returns_empty(self, tmp_path):
        from src.evaluation.dataset import filter_cases, load_golden_dataset

        cases = load_golden_dataset(_write_dataset(tmp_path, SYNTHETIC_ENTRIES))

        assert filter_cases(cases, category="edge_case") == []


class TestGoldenCase:
    def test_is_pydantic_model(self, tmp_path):
        from pydantic import BaseModel

        from src.evaluation.dataset import GoldenCase

        assert issubclass(GoldenCase, BaseModel)
