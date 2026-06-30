class TestVectorStoreUpsert:
    def test_upsert_stores_chunks_and_returns_ids(self, settings, make_chunk):
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

    def test_upsert_stores_expected_metadata(self, settings, make_chunk):
        import chromadb

        from src.retrieval.vector_store import COLLECTION_NAME, VectorStore

        vs = VectorStore(settings)
        vs.upsert([make_chunk(0, text="hello world")], [[1.0, 0.0, 0.0]])

        col = chromadb.PersistentClient(path=settings.chroma_persist_dir_str).get_collection(COLLECTION_NAME)
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

    def test_upsert_is_idempotent(self, settings, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings)
        chunk = make_chunk(0)
        vs.upsert([chunk], [[1.0, 0.0, 0.0]])
        vs.upsert([chunk], [[1.0, 0.0, 0.0]])

        assert vs.count() == 1


class TestVectorStoreFilterDuplicates:
    def test_all_accepted_when_collection_empty(self, settings, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings)
        chunks = [make_chunk(0), make_chunk(1)]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        accepted_chunks, accepted_embeddings = vs.filter_duplicates(chunks, embeddings)

        assert accepted_chunks == chunks
        assert accepted_embeddings == embeddings

    def test_near_duplicate_excluded(self, settings, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings)
        vs.upsert([make_chunk(0)], [[1.0, 0.0, 0.0]])

        accepted, _ = vs.filter_duplicates([make_chunk(1)], [[1.0, 0.0, 0.0]])

        assert accepted == []

    def test_dissimilar_chunk_accepted(self, settings, make_chunk):
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings)
        vs.upsert([make_chunk(0)], [[1.0, 0.0, 0.0]])

        accepted, _ = vs.filter_duplicates([make_chunk(1)], [[0.0, 1.0, 0.0]])

        assert accepted == [make_chunk(1)]

    def test_duplicate_flagged_in_logs(self, settings, make_chunk, caplog):
        import logging
        from unittest.mock import patch
        from src.retrieval.vector_store import VectorStore

        vs = VectorStore(settings)
        vs.upsert([make_chunk(0)], [[1.0, 0.0, 0.0]])

        # Patch query to return distance 0.0 (similarity 1.0 → duplicate)
        mock_results = {"distances": [[0.0]]}
        with patch.object(vs._collection, "query", return_value=mock_results):
            with caplog.at_level(logging.DEBUG, logger="src.retrieval.vector_store"):
                accepted, _ = vs.filter_duplicates([make_chunk(1)], [[1.0, 0.0, 0.0]])

        assert accepted == []
        assert any("duplicate" in r.message.lower() for r in caplog.records)
