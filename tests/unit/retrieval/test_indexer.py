from unittest.mock import MagicMock


class TestIndexerIndex:
    def test_index_returns_stored_chunk_ids(self, settings, make_chunk):
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

    def test_index_stores_in_both_indexes(self, settings, make_chunk):
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

    def test_index_excludes_duplicates_from_both_stores(self, settings, make_chunk):
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
        from src.retrieval.bm25_store import BM25Store
        from src.retrieval.embedder import Embedder
        from src.retrieval.indexer import Indexer
        from src.retrieval.vector_store import VectorStore

        assert Indexer(
            settings,
            embedder=MagicMock(spec=Embedder),
            vector_store=MagicMock(spec=VectorStore),
            bm25_store=MagicMock(spec=BM25Store),
        ).index([]) == []

    def test_index_persists_bm25_to_disk(self, settings, make_chunk):
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
