class TestVectorStoreUpsert:
    def test_upsert_stores_chunks_and_returns_ids(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        chunks = [make_chunk(0), make_chunk(1)]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        stored = vs.upsert(chunks, embeddings)

        assert stored == ["chunk-000", "chunk-001"]
        assert vs.count() == 2

    def test_upsert_empty_list_returns_empty(self, settings, embedder):
        from src.retrieval.vector_store import VectorStore

        assert VectorStore(settings, embedder).upsert([], []) == []

    def test_upsert_stores_expected_metadata(self, settings, embedder, make_chunk):
        import chromadb

        from src.retrieval.vector_store import COLLECTION_NAME, VectorStore

        vs = VectorStore(settings, embedder)
        vs.upsert([make_chunk(0, text="hello world")], [[1.0, 0.0, 0.0]])

        col = chromadb.PersistentClient(
            path=settings.chroma_persist_dir_str
        ).get_collection(COLLECTION_NAME)
        metadatas = col.get(ids=["chunk-000"], include=["metadatas"])["metadatas"]
        assert metadatas is not None
        meta = metadatas[0]

        assert meta["source_path"] == "/data/doc-000.md"
        assert meta["chunk_index"] == 0
        assert meta["section_heading"] == ""
        assert meta["strategy"] == "fixed_size"
        assert meta["char_count"] == len("hello world")
        assert meta["doc_id"] == "doc-000"
        assert meta["title"] == "Doc 0"

    def test_upsert_is_idempotent(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        chunk = make_chunk(0)
        vs.upsert([chunk], [[1.0, 0.0, 0.0]])
        vs.upsert([chunk], [[1.0, 0.0, 0.0]])

        assert vs.count() == 1


class TestVectorStoreFilterDuplicates:
    def test_all_accepted_when_collection_empty(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        chunks = [make_chunk(0), make_chunk(1)]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        accepted_chunks, accepted_embeddings = vs.filter_duplicates(chunks, embeddings)

        assert accepted_chunks == chunks
        assert accepted_embeddings == embeddings

    def test_near_duplicate_excluded(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        vs.upsert([make_chunk(0)], [[1.0, 0.0, 0.0]])

        accepted, _ = vs.filter_duplicates([make_chunk(1)], [[1.0, 0.0, 0.0]])

        assert accepted == []

    def test_dissimilar_chunk_accepted(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        vs.upsert([make_chunk(0)], [[1.0, 0.0, 0.0]])

        accepted, _ = vs.filter_duplicates([make_chunk(1)], [[0.0, 1.0, 0.0]])

        assert accepted == [make_chunk(1)]

    def test_duplicate_flagged_in_logs(self, settings, embedder, make_chunk, caplog):
        import logging
        from unittest.mock import patch

        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        vs.upsert([make_chunk(0)], [[1.0, 0.0, 0.0]])

        # Patch query to return distance 0.0 (similarity 1.0 → duplicate)
        mock_results = {"distances": [[0.0]]}
        with patch.object(vs._collection, "query", return_value=mock_results), caplog.at_level(logging.DEBUG, logger="src.retrieval.vector_store"):
            accepted, _ = vs.filter_duplicates([make_chunk(1)], [[1.0, 0.0, 0.0]])

        assert accepted == []
        assert any("duplicate" in r.message.lower() for r in caplog.records)


class TestVectorStoreQuery:
    def test_query_returns_hits_sorted_by_similarity(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        chunks = [
            make_chunk(0, text="alpha"),
            make_chunk(1, text="beta"),
            make_chunk(2, text="gamma"),
        ]
        vs.upsert(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        hits = vs.query([1.0, 0.0, 0.0], k=3)

        assert len(hits) == 3
        assert hits[0].chunk_id == "chunk-000"
        assert hits[0].similarity > hits[1].similarity >= hits[2].similarity

    def test_query_empty_collection_returns_empty(self, settings, embedder):
        from src.retrieval.vector_store import VectorStore

        hits = VectorStore(settings, embedder).query([1.0, 0.0, 0.0], k=10)
        assert hits == []

    def test_query_k_exceeds_count_returns_all(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        vs.upsert([make_chunk(0)], [[1.0, 0.0, 0.0]])

        hits = vs.query([1.0, 0.0, 0.0], k=10)

        assert len(hits) == 1

    def test_query_limits_results_to_k(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        chunks = [make_chunk(i) for i in range(5)]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.7, 0.7, 0.0],
            [0.0, 0.7, 0.7],
        ]
        vs.upsert(chunks, embeddings)

        hits = vs.query([1.0, 0.0, 0.0], k=3)

        assert len(hits) == 3

    def test_query_hit_fields_populated(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        chunk = make_chunk(0, text="hello world")
        vs.upsert([chunk], [[1.0, 0.0, 0.0]])

        hit = vs.query([1.0, 0.0, 0.0], k=1)[0]

        assert hit.chunk_id == "chunk-000"
        assert hit.text == "hello world"
        assert hit.doc_id == "doc-000"
        assert hit.source_path == "/data/doc-000.md"
        assert hit.title == "Doc 0"
        assert hit.section_heading is None  # stored as "" → converted back to None
        assert hit.chunk_index == 0
        assert hit.strategy == "fixed_size"
        assert 0.0 <= hit.similarity <= 1.0


class TestVectorStoreGetByIds:
    def test_get_by_ids_empty_list_returns_empty(self, settings, embedder):
        from src.retrieval.vector_store import VectorStore

        assert VectorStore(settings, embedder).get_by_ids([]) == []

    def test_get_by_ids_returns_hit_with_correct_fields(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        chunk = make_chunk(0, text="hello world")
        vs.upsert([chunk], [[1.0, 0.0, 0.0]])

        hits = vs.get_by_ids(["chunk-000"])

        assert len(hits) == 1
        hit = hits[0]
        assert hit.chunk_id == "chunk-000"
        assert hit.text == "hello world"
        assert hit.doc_id == "doc-000"
        assert hit.source_path == "/data/doc-000.md"
        assert hit.title == "Doc 0"
        assert hit.section_heading is None
        assert hit.chunk_index == 0
        assert hit.strategy == "fixed_size"

    def test_get_by_ids_similarity_is_zero_sentinel(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        vs.upsert([make_chunk(0)], [[1.0, 0.0, 0.0]])

        hit = vs.get_by_ids(["chunk-000"])[0]

        assert hit.similarity == 0.0

    def test_get_by_ids_multiple_ids(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        chunks = [make_chunk(0, text="alpha"), make_chunk(1, text="beta")]
        vs.upsert(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        hits = vs.get_by_ids(["chunk-000", "chunk-001"])

        chunk_ids = {h.chunk_id for h in hits}
        assert chunk_ids == {"chunk-000", "chunk-001"}

    def test_get_by_ids_missing_id_omitted(self, settings, embedder, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings, embedder)
        vs.upsert([make_chunk(0)], [[1.0, 0.0, 0.0]])

        hits = vs.get_by_ids(["chunk-000", "chunk-999"])

        assert len(hits) == 1
        assert hits[0].chunk_id == "chunk-000"
