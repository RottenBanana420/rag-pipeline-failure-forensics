"""End-to-end ingestion roundtrip: load → save → load_processed → verify."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.ingestion.loader import DocumentLoader
from src.ingestion.storage import load_processed, save_processed
from tests.fixtures.create_pdf import create_sample_pdf

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture()
def loader() -> DocumentLoader:
    return DocumentLoader(Settings())


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pdf"
    create_sample_pdf(p)
    return p


@pytest.fixture()
def processed_dir(tmp_path: Path) -> Path:
    d = tmp_path / "processed"
    d.mkdir()
    return d


class TestIngestionRoundtrip:
    @pytest.mark.parametrize("fixture_name", ["sample.md", "sample.txt", "sample.html"])
    def test_roundtrip_text_formats(
        self,
        fixture_name: str,
        loader: DocumentLoader,
        processed_dir: Path,
    ):
        raw_path = FIXTURES / fixture_name
        docs = loader.load(raw_path)
        assert len(docs) > 0, f"{fixture_name} produced no documents"

        save_processed(docs, raw_path, processed_dir)
        restored = load_processed(raw_path, processed_dir)

        assert len(restored) == len(docs)
        for original, loaded in zip(docs, restored, strict=True):
            assert original.doc_id == loaded.doc_id
            assert original.source_format == loaded.source_format
            assert original.title == loaded.title
            assert original.section_heading == loaded.section_heading
            assert original.page_number == loaded.page_number
            assert original.text == loaded.text

    def test_roundtrip_pdf(
        self,
        sample_pdf: Path,
        loader: DocumentLoader,
        processed_dir: Path,
    ):
        docs = loader.load(sample_pdf)
        assert len(docs) == 2

        save_processed(docs, sample_pdf, processed_dir)
        restored = load_processed(sample_pdf, processed_dir)

        assert len(restored) == 2
        for original, loaded in zip(docs, restored, strict=True):
            assert original == loaded

    def test_reingest_produces_identical_output(
        self,
        loader: DocumentLoader,
        processed_dir: Path,
    ):
        raw_path = FIXTURES / "sample.md"

        docs1 = loader.load(raw_path)
        save_processed(docs1, raw_path, processed_dir)
        restored1 = load_processed(raw_path, processed_dir)

        docs2 = loader.load(raw_path)
        save_processed(docs2, raw_path, processed_dir)
        restored2 = load_processed(raw_path, processed_dir)

        def without_timestamp(docs):
            return [d.model_dump(exclude={"processed_at"}) for d in docs]

        assert without_timestamp(restored1) == without_timestamp(restored2)

    def test_all_formats_produce_non_empty_text(
        self,
        loader: DocumentLoader,
        sample_pdf: Path,
    ):
        fixtures = [
            FIXTURES / "sample.md",
            FIXTURES / "sample.txt",
            FIXTURES / "sample.html",
            sample_pdf,
        ]
        for raw_path in fixtures:
            docs = loader.load(raw_path)
            assert all(d.text.strip() for d in docs), f"Empty text in docs from {raw_path.name}"
