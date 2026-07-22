"""Loader for the Phase 6 golden Q&A dataset (data/golden/qa_dataset.json).

Path resolution mirrors `tests/unit/evaluation/test_golden_dataset.py`'s own
`GOLDEN_DIR`/`DATASET_PATH` constants.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden"
CORPUS_DIR = GOLDEN_DIR / "corpus"
DATASET_PATH = GOLDEN_DIR / "qa_dataset.json"

VALID_CATEGORIES = frozenset(
    {"lookup", "multi_hop", "no_answer", "ambiguous", "edge_case"}
)


class GoldenCase(BaseModel):
    id: str
    question: str
    expected_answer: str
    category: str
    source_documents: list[str]
    source_sections: list[str | None]
    notes: str | None = None


def load_golden_dataset(path: Path = DATASET_PATH) -> list[GoldenCase]:
    entries = json.loads(path.read_text())
    return [GoldenCase.model_validate(entry) for entry in entries]


def filter_cases(
    cases: list[GoldenCase],
    category: str | None = None,
    limit: int | None = None,
) -> list[GoldenCase]:
    filtered = (
        cases if category is None else [c for c in cases if c.category == category]
    )
    return filtered if limit is None else filtered[:limit]
