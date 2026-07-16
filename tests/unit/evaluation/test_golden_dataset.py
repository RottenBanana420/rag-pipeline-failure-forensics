"""Structural integrity checks for the Phase 6 golden Q&A dataset.

Guards data/golden/qa_dataset.json against silent drift (bad category, duplicate
id, dangling document/section reference) as entries are hand-edited or, later,
auto-appended from confirmed production flags (see docs/DECISIONS.md).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "data" / "golden"
CORPUS_DIR = GOLDEN_DIR / "corpus"
DATASET_PATH = GOLDEN_DIR / "qa_dataset.json"

VALID_CATEGORIES = {"lookup", "multi_hop", "no_answer", "ambiguous", "edge_case"}


@pytest.fixture(scope="module")
def entries() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(DATASET_PATH.read_text())
    return data


@pytest.fixture(scope="module")
def corpus_headings() -> dict[str, set[str]]:
    headings = {}
    for path in CORPUS_DIR.glob("*.md"):
        found = re.findall(r"^##\s+(.+)$", path.read_text(), re.MULTILINE)
        headings[path.name] = {h.strip() for h in found}
    return headings


def test_has_at_least_fifty_entries(entries: list[dict[str, Any]]) -> None:
    assert len(entries) >= 50


def test_ids_are_unique(entries: list[dict[str, Any]]) -> None:
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))


def test_ids_follow_qa_prefix_convention(entries: list[dict[str, Any]]) -> None:
    for e in entries:
        assert re.fullmatch(r"qa-\d{3,}", e["id"]), e["id"]


def test_categories_are_from_the_closed_set(entries: list[dict[str, Any]]) -> None:
    for e in entries:
        assert e["category"] in VALID_CATEGORIES, e["id"]


def test_every_category_is_represented(entries: list[dict[str, Any]]) -> None:
    seen = {e["category"] for e in entries}
    assert seen == VALID_CATEGORIES


def test_question_and_expected_answer_are_nonempty(
    entries: list[dict[str, Any]],
) -> None:
    for e in entries:
        assert e["question"].strip(), e["id"]
        assert e["expected_answer"].strip(), e["id"]


def test_source_documents_resolve_to_real_corpus_files(
    entries: list[dict[str, Any]], corpus_headings: dict[str, set[str]]
) -> None:
    for e in entries:
        for doc in e["source_documents"]:
            assert doc in corpus_headings, f"{e['id']} references missing doc {doc}"


def test_source_sections_resolve_to_real_headings(
    entries: list[dict[str, Any]], corpus_headings: dict[str, set[str]]
) -> None:
    for e in entries:
        docs = e["source_documents"]
        for section in e["source_sections"]:
            assert any(section in corpus_headings[doc] for doc in docs), (
                f"{e['id']}: section {section!r} not found in {docs}"
            )


def test_no_answer_entries_have_no_source_references(
    entries: list[dict[str, Any]],
) -> None:
    for e in entries:
        if e["category"] == "no_answer":
            assert e["source_documents"] == []
            assert e["source_sections"] == []
