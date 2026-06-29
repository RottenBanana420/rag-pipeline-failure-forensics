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


class TestBM25StoreAdd:
    def test_add_increments_count(self, settings):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.add([make_chunk(0), make_chunk(1)])
        assert store.count() == 2

    def test_add_is_cumulative(self, settings):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.add([make_chunk(0)])
        store.add([make_chunk(1)])
        assert store.count() == 2

    def test_add_empty_list_is_noop(self, settings):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.add([])
        assert store.count() == 0


class TestBM25StoreGetScores:
    def test_get_scores_returns_pair_per_chunk(self, settings):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.add([make_chunk(0, "hello world"), make_chunk(1, "foo bar")])
        scores = store.get_scores("hello world")

        assert len(scores) == 2
        assert set(pair[0] for pair in scores) == {"chunk-000", "chunk-001"}

    def test_get_scores_ranks_relevant_chunk_higher(self, settings):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.add([make_chunk(0, "hello world"), make_chunk(1, "foo bar"), make_chunk(2, "unrelated content here")])
        score_map = dict(store.get_scores("hello world"))

        assert score_map["chunk-000"] > score_map["chunk-001"]

    def test_get_scores_empty_store_returns_empty(self, settings):
        from src.retrieval.bm25_store import BM25Store

        assert BM25Store(settings).get_scores("anything") == []


class TestBM25StorePersistence:
    def test_save_and_load_restores_index(self, settings):
        from src.retrieval.bm25_store import BM25Store

        store1 = BM25Store(settings)
        store1.add([make_chunk(0, "hello world"), make_chunk(1, "foo bar"), make_chunk(2, "unrelated content here")])
        store1.save()

        store2 = BM25Store(settings)
        store2.load()

        assert store2.count() == 3
        score_map = dict(store2.get_scores("hello world"))
        assert score_map["chunk-000"] > score_map["chunk-001"]

    def test_load_on_missing_file_is_noop(self, settings):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.load()
        assert store.count() == 0
