class TestBM25StoreAdd:
    def test_add_increments_count(self, settings, make_chunk):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.add([make_chunk(0), make_chunk(1)])
        assert store.count() == 2

    def test_add_is_cumulative(self, settings, make_chunk):
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
    def test_get_scores_returns_pair_per_chunk(self, settings, make_chunk):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.add([make_chunk(0, "hello world"), make_chunk(1, "foo bar")])
        scores = store.get_scores("hello world")

        assert len(scores) == 2
        assert set(pair[0] for pair in scores) == {"chunk-000", "chunk-001"}

    def test_get_scores_ranks_relevant_chunk_higher(self, settings, make_chunk):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.add([make_chunk(0, "hello world"), make_chunk(1, "foo bar"), make_chunk(2, "unrelated content here")])
        score_map = dict(store.get_scores("hello world"))

        assert score_map["chunk-000"] > score_map["chunk-001"]

    def test_get_scores_empty_store_returns_empty(self, settings):
        from src.retrieval.bm25_store import BM25Store

        assert BM25Store(settings).get_scores("anything") == []

    def test_get_scores_matches_hyphenated_term_in_text(self, settings, make_chunk):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.add([
            make_chunk(0, "config-key: required value"),
            make_chunk(1, "unrelated content here"),
            make_chunk(2, "something else entirely"),
        ])
        scores = dict(store.get_scores("config key"))

        assert scores["chunk-000"] > 0.0
        assert scores["chunk-001"] == 0.0

    def test_get_scores_matches_punctuated_query(self, settings, make_chunk):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.add([
            make_chunk(0, "error code 404 not found"),
            make_chunk(1, "success response ok"),
            make_chunk(2, "unrelated content here"),
        ])
        scores = dict(store.get_scores("error-code-404"))

        assert scores["chunk-000"] > 0.0
        assert scores["chunk-001"] == 0.0

    def test_get_scores_preserves_underscore_identifiers(self, settings, make_chunk):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.add([
            make_chunk(0, "set OPENAI_API_KEY in env"),
            make_chunk(1, "other config values"),
            make_chunk(2, "unrelated content here"),
        ])
        scores = dict(store.get_scores("openai_api_key"))

        assert scores["chunk-000"] > 0.0


class TestBM25StoreLazyRebuild:
    def test_get_scores_works_after_sequential_adds(self, settings, make_chunk):
        from src.retrieval.bm25_store import BM25Store

        store = BM25Store(settings)
        store.add([make_chunk(0, "hello world")])
        store.add([make_chunk(1, "foo bar baz")])
        store.add([make_chunk(2, "unrelated stuff")])
        scores = dict(store.get_scores("hello world"))

        assert scores["chunk-000"] > scores["chunk-001"]
        assert scores["chunk-000"] > scores["chunk-002"]

    def test_get_scores_after_load_then_add(self, settings, make_chunk):
        from src.retrieval.bm25_store import BM25Store

        store1 = BM25Store(settings)
        store1.add([make_chunk(0, "hello world"), make_chunk(1, "unrelated content here")])
        store1.save()

        store2 = BM25Store(settings)
        store2.load()
        store2.add([make_chunk(2, "foo bar baz")])
        scores = dict(store2.get_scores("hello world"))

        assert scores["chunk-000"] > scores["chunk-002"]


class TestBM25StorePersistence:
    def test_save_and_load_restores_index(self, settings, make_chunk):
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
