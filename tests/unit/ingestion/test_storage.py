from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.ingestion.loader import DocumentLoader
from src.ingestion.storage import list_raw_files, load_processed, save_processed
from tests.fixtures.create_pdf import create_sample_pdf

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


@pytest.fixture()
def loader() -> DocumentLoader:
    return DocumentLoader(Settings())


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pdf"
    create_sample_pdf(p)
    return p


class TestSaveProcessed:
    def test_creates_output_directory(self, loader: DocumentLoader, tmp_path: Path):
        docs = loader.load(FIXTURES / "sample.txt")
        raw_path = FIXTURES / "sample.txt"
        processed_dir = tmp_path / "processed"
        save_processed(docs, raw_path, processed_dir)
        out_dir = processed_dir / "sample.txt"
        assert out_dir.is_dir()

    def test_writes_one_json_per_doc(self, loader: DocumentLoader, tmp_path: Path):
        docs = loader.load(FIXTURES / "sample.txt")
        raw_path = FIXTURES / "sample.txt"
        processed_dir = tmp_path / "processed"
        save_processed(docs, raw_path, processed_dir)
        out_dir = processed_dir / "sample.txt"
        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) == len(docs)

    def test_pdf_files_named_by_page(self, loader: DocumentLoader, tmp_path: Path, sample_pdf: Path):
        docs = loader.load(sample_pdf)
        processed_dir = tmp_path / "processed"
        save_processed(docs, sample_pdf, processed_dir)
        out_dir = processed_dir / sample_pdf.name
        assert (out_dir / "page_001.json").exists()
        assert (out_dir / "page_002.json").exists()

    def test_non_pdf_files_named_by_section(self, loader: DocumentLoader, tmp_path: Path):
        docs = loader.load(FIXTURES / "sample.md")
        raw_path = FIXTURES / "sample.md"
        processed_dir = tmp_path / "processed"
        save_processed(docs, raw_path, processed_dir)
        out_dir = processed_dir / "sample.md"
        json_files = sorted(out_dir.glob("*.json"))
        assert all(f.name.startswith("section_") for f in json_files)

    def test_overwrite_on_reingest(self, loader: DocumentLoader, tmp_path: Path):
        docs = loader.load(FIXTURES / "sample.txt")
        raw_path = FIXTURES / "sample.txt"
        processed_dir = tmp_path / "processed"
        save_processed(docs, raw_path, processed_dir)
        save_processed(docs, raw_path, processed_dir)  # second call
        out_dir = processed_dir / "sample.txt"
        assert len(list(out_dir.glob("*.json"))) == len(docs)


class TestLoadProcessed:
    def test_round_trips_all_fields(self, loader: DocumentLoader, tmp_path: Path):
        docs = loader.load(FIXTURES / "sample.md")
        raw_path = FIXTURES / "sample.md"
        processed_dir = tmp_path / "processed"
        save_processed(docs, raw_path, processed_dir)

        restored = load_processed(raw_path, processed_dir)
        assert len(restored) == len(docs)
        for original, loaded in zip(docs, restored, strict=True):
            assert original == loaded

    def test_returns_empty_list_if_not_found(self, tmp_path: Path):
        raw_path = tmp_path / "nonexistent.txt"
        processed_dir = tmp_path / "processed"
        assert load_processed(raw_path, processed_dir) == []


class TestListRawFiles:
    def test_yields_supported_extensions(self, tmp_path: Path):
        for name in ["a.md", "b.txt", "c.html", "d.pdf", "e.docx", "f.csv"]:
            (tmp_path / name).write_bytes(b"x")
        found = set(p.name for p in list_raw_files(tmp_path))
        assert found == {"a.md", "b.txt", "c.html", "d.pdf"}

    def test_recurses_into_subdirectories(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("# Title\nContent")
        found = list(list_raw_files(tmp_path))
        assert any(p.name == "nested.md" for p in found)

    def test_empty_directory_yields_nothing(self, tmp_path: Path):
        assert list(list_raw_files(tmp_path)) == []
