from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.ingestion.loader import DocumentLoader
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


class TestMarkdownLoader:
    def test_returns_one_doc_per_section(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.md")
        # sample.md has: intro section (before first heading) may be empty,
        # h1 "Getting Started", h2 "Installation", h2 "Configuration"
        assert len(docs) >= 2

    def test_section_headings_populated(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.md")
        headings = [d.section_heading for d in docs]
        assert "Installation" in headings
        assert "Configuration" in headings

    def test_no_page_numbers(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.md")
        assert all(d.page_number is None for d in docs)

    def test_source_format_is_markdown(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.md")
        assert all(d.source_format == "markdown" for d in docs)

    def test_no_markdown_markup_in_text(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.md")
        full_text = " ".join(d.text for d in docs)
        assert "```" not in full_text
        assert "##" not in full_text

    def test_doc_id_stable_across_loads(self, loader: DocumentLoader):
        docs1 = loader.load(FIXTURES / "sample.md")
        docs2 = loader.load(FIXTURES / "sample.md")
        assert docs1[0].doc_id == docs2[0].doc_id

    def test_all_docs_share_same_doc_id(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.md")
        ids = {d.doc_id for d in docs}
        assert len(ids) == 1

    def test_title_is_first_heading(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.md")
        assert docs[0].title == "Getting Started"

    def test_empty_file_returns_empty_list(self, loader: DocumentLoader, tmp_path: Path):
        empty = tmp_path / "empty.md"
        empty.write_text("")
        assert loader.load(empty) == []


class TestTextLoader:
    def test_returns_single_doc(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.txt")
        assert len(docs) == 1

    def test_section_heading_is_none(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.txt")
        assert docs[0].section_heading is None

    def test_page_number_is_none(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.txt")
        assert docs[0].page_number is None

    def test_source_format_is_text(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.txt")
        assert docs[0].source_format == "text"

    def test_full_content_present(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.txt")
        assert "plain text document" in docs[0].text

    def test_title_is_filename_stem(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.txt")
        assert docs[0].title == "sample"

    def test_empty_file_returns_empty_list(self, loader: DocumentLoader, tmp_path: Path):
        empty = tmp_path / "empty.txt"
        empty.write_text("   \n  ")
        assert loader.load(empty) == []


class TestHtmlLoader:
    def test_returns_one_doc_per_section(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.html")
        # sample.html has h1 "Overview" and h2 "Components"
        assert len(docs) >= 2

    def test_section_headings_populated(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.html")
        headings = [d.section_heading for d in docs]
        assert "Overview" in headings
        assert "Components" in headings

    def test_no_html_tags_in_text(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.html")
        for doc in docs:
            assert "<" not in doc.text
            assert ">" not in doc.text

    def test_script_content_stripped(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.html")
        full_text = " ".join(d.text for d in docs)
        assert "console.log" not in full_text

    def test_source_format_is_html(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.html")
        assert all(d.source_format == "html" for d in docs)

    def test_title_from_h1(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.html")
        assert docs[0].title == "Sample HTML Document"

    def test_no_page_numbers(self, loader: DocumentLoader):
        docs = loader.load(FIXTURES / "sample.html")
        assert all(d.page_number is None for d in docs)


class TestPdfLoader:
    def test_returns_one_doc_per_page(self, loader: DocumentLoader, sample_pdf: Path):
        docs = loader.load(sample_pdf)
        assert len(docs) == 2

    def test_page_numbers_are_one_indexed(self, loader: DocumentLoader, sample_pdf: Path):
        docs = loader.load(sample_pdf)
        page_nums = [d.page_number for d in docs]
        assert page_nums == [1, 2]

    def test_section_heading_is_none(self, loader: DocumentLoader, sample_pdf: Path):
        docs = loader.load(sample_pdf)
        assert all(d.section_heading is None for d in docs)

    def test_source_format_is_pdf(self, loader: DocumentLoader, sample_pdf: Path):
        docs = loader.load(sample_pdf)
        assert all(d.source_format == "pdf" for d in docs)

    def test_text_content_extracted(self, loader: DocumentLoader, sample_pdf: Path):
        docs = loader.load(sample_pdf)
        assert "Page one" in docs[0].text
        assert "Page two" in docs[1].text

    def test_doc_id_stable(self, loader: DocumentLoader, sample_pdf: Path):
        docs1 = loader.load(sample_pdf)
        docs2 = loader.load(sample_pdf)
        assert docs1[0].doc_id == docs2[0].doc_id

    def test_all_docs_share_doc_id(self, loader: DocumentLoader, sample_pdf: Path):
        docs = loader.load(sample_pdf)
        assert len({d.doc_id for d in docs}) == 1


class TestLoaderDispatch:
    def test_unsupported_extension_raises_value_error(self, loader: DocumentLoader, tmp_path: Path):
        f = tmp_path / "file.docx"
        f.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            loader.load(f)

    def test_htm_extension_treated_as_html(self, loader: DocumentLoader, tmp_path: Path):
        htm = tmp_path / "page.htm"
        htm.write_text("<html><body><h1>Title</h1><p>Body text.</p></body></html>")
        docs = loader.load(htm)
        assert all(d.source_format == "html" for d in docs)
