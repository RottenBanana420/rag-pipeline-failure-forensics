import pytest

from src.ingestion import Chunk


@pytest.fixture
def make_chunk():
    def _factory(idx: int, text: str = "sample text") -> Chunk:
        return Chunk(
            chunk_id=f"chunk-{idx:03d}",
            doc_id=f"doc-{idx:03d}",
            source_path=f"/data/doc-{idx:03d}.md",
            source_format="markdown",
            title=f"Doc {idx}",
            section_heading=None,
            page_number=None,
            text=text,
            chunk_index=idx,
            strategy="fixed_size",
            processed_at="2024-01-01T00:00:00Z",
        )

    return _factory


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()
