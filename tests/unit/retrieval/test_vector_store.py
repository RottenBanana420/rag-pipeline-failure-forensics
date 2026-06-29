import pytest

from src.ingestion import Chunk


def make_chunk(idx: int, text: str = "sample text") -> Chunk:
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


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings
    return Settings()


class TestVectorStoreUpsert:
    def test_upsert_stores_chunks_and_returns_ids(self, settings):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings)
        chunks = [make_chunk(0), make_chunk(1)]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        stored = vs.upsert(chunks, embeddings)

        assert stored == ["chunk-000", "chunk-001"]
        assert vs.count() == 2

    def test_upsert_empty_list_returns_empty(self, settings):
        from src.retrieval.vector_store import VectorStore

        assert VectorStore(settings).upsert([], []) == []

    def test_upsert_stores_expected_metadata(self, settings):
        import chromadb

        from src.retrieval.vector_store import COLLECTION_NAME, VectorStore

        vs = VectorStore(settings)
        vs.upsert([make_chunk(0, text="hello world")], [[1.0, 0.0, 0.0]])

        col = chromadb.PersistentClient(path=settings.chroma_persist_dir_str).get_collection(COLLECTION_NAME)
        meta = col.get(ids=["chunk-000"], include=["metadatas"])["metadatas"][0]

        assert meta["source_path"] == "/data/doc-000.md"
        assert meta["chunk_index"] == 0
        assert meta["section_heading"] == ""
        assert meta["strategy"] == "fixed_size"
        assert meta["char_count"] == len("hello world")

    def test_upsert_is_idempotent(self, settings):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings)
        chunk = make_chunk(0)
        vs.upsert([chunk], [[1.0, 0.0, 0.0]])
        vs.upsert([chunk], [[1.0, 0.0, 0.0]])

        assert vs.count() == 1


class TestVectorStoreFilterDuplicates:
    def test_all_accepted_when_collection_empty(self, settings):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings)
        chunks = [make_chunk(0), make_chunk(1)]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        accepted_chunks, accepted_embeddings = vs.filter_duplicates(chunks, embeddings)

        assert accepted_chunks == chunks
        assert accepted_embeddings == embeddings

    def test_near_duplicate_excluded(self, settings):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings)
        vs.upsert([make_chunk(0)], [[1.0, 0.0, 0.0]])

        accepted, _ = vs.filter_duplicates([make_chunk(1)], [[1.0, 0.0, 0.0]])

        assert accepted == []

    def test_dissimilar_chunk_accepted(self, settings):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings)
        vs.upsert([make_chunk(0)], [[1.0, 0.0, 0.0]])

        accepted, _ = vs.filter_duplicates([make_chunk(1)], [[0.0, 1.0, 0.0]])

        assert accepted == [make_chunk(1)]
