from unittest.mock import MagicMock

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


class TestIndexerIndex:
    def test_index_returns_stored_chunk_ids(self, settings):
        from src.retrieval.bm25_store import BM25Store
        from src.retrieval.embedder import Embedder
        from src.retrieval.indexer import Indexer
        from src.retrieval.vector_store import VectorStore

        chunks = [make_chunk(0), make_chunk(1), make_chunk(2)]
        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = [[float(i)] * 3 for i in range(3)]

        stored = Indexer(
            settings,
            embedder=embedder,
            vector_store=VectorStore(settings),
            bm25_store=BM25Store(settings),
        ).index(chunks)

        assert stored == ["chunk-000", "chunk-001", "chunk-002"]

    def test_index_stores_in_both_indexes(self, settings):
        from src.retrieval.bm25_store import BM25Store
        from src.retrieval.embedder import Embedder
        from src.retrieval.indexer import Indexer
        from src.retrieval.vector_store import VectorStore

        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        vector_store = VectorStore(settings)
        bm25_store = BM25Store(settings)

        Indexer(settings, embedder=embedder, vector_store=vector_store, bm25_store=bm25_store).index(
            [make_chunk(0), make_chunk(1)]
        )

        assert vector_store.count() == 2
        assert bm25_store.count() == 2

    def test_index_excludes_duplicates_from_both_stores(self, settings):
        from src.retrieval.bm25_store import BM25Store
        from src.retrieval.embedder import Embedder
        from src.retrieval.indexer import Indexer
        from src.retrieval.vector_store import VectorStore

        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = [[1.0, 0.0, 0.0]]

        vector_store = VectorStore(settings)
        bm25_store = BM25Store(settings)
        indexer = Indexer(settings, embedder=embedder, vector_store=vector_store, bm25_store=bm25_store)

        indexer.index([make_chunk(0)])
        stored = indexer.index([make_chunk(1)])  # same embedding → duplicate

        assert stored == []
        assert vector_store.count() == 1
        assert bm25_store.count() == 1

    def test_index_empty_list_returns_empty(self, settings):
        from src.retrieval.indexer import Indexer

        assert Indexer(settings).index([]) == []

    def test_index_persists_bm25_to_disk(self, settings):
        from src.retrieval.bm25_store import BM25Store
        from src.retrieval.embedder import Embedder
        from src.retrieval.indexer import Indexer
        from src.retrieval.vector_store import VectorStore

        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = [[1.0, 0.0, 0.0]]

        Indexer(
            settings,
            embedder=embedder,
            vector_store=VectorStore(settings),
            bm25_store=BM25Store(settings),
        ).index([make_chunk(0, text="python search engine")])

        restored = BM25Store(settings)
        restored.load()
        scores = restored.get_scores("python search")

        assert len(scores) == 1
        assert scores[0][0] == "chunk-000"
