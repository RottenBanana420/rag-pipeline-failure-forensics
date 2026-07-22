"""Unit tests for src/evaluation/corpus.py (ensure_golden_corpus_indexed).

Uses a fake embedder (mirrors tests/unit/retrieval/conftest.py's `embedder`
fixture convention) so tests don't load a real sentence_transformers model —
DocumentLoader/Chunker/Indexer/ChromaVectorStore are exercised for real
against a tmp_path-isolated Chroma directory.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_embedder():
    from src.retrieval.embedder import EmbedderProtocol

    mock = MagicMock(spec=EmbedderProtocol)
    mock.provider_id = "test/fake-embedder"
    mock.dimensions = 3
    mock.embed.side_effect = lambda texts: [
        [float(i), 0.0, 0.0] for i in range(len(texts))
    ]
    return mock


@pytest.fixture
def eval_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("EVAL_CHROMA_PERSIST_DIR", str(tmp_path / "eval-chroma"))
    from src.config import Settings

    base = Settings()
    return base.model_copy(update={"chroma_persist_dir": base.eval_chroma_persist_dir})


@pytest.fixture
def tiny_corpus_dir(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc-one.md").write_text(
        "# Doc One\n\n## Section A\n\nSome content about topic A.\n"
    )
    (corpus / "doc-two.md").write_text(
        "# Doc Two\n\n## Section B\n\nSome content about topic B.\n"
    )
    return corpus


class TestEnsureGoldenCorpusIndexed:
    def test_indexes_corpus_when_store_is_empty(
        self, eval_settings, fake_embedder, tiny_corpus_dir, monkeypatch
    ):
        from src.evaluation import corpus
        from src.retrieval.vector_store import make_vector_store

        monkeypatch.setattr(corpus, "make_embedder", lambda settings: fake_embedder)

        corpus.ensure_golden_corpus_indexed(eval_settings, corpus_dir=tiny_corpus_dir)

        vector_store = make_vector_store(eval_settings, fake_embedder)
        assert vector_store.count() > 0

    def test_idempotent_skips_reindexing_when_already_populated(
        self, eval_settings, fake_embedder, tiny_corpus_dir, monkeypatch
    ):
        from src.evaluation import corpus
        from src.retrieval.vector_store import make_vector_store

        monkeypatch.setattr(corpus, "make_embedder", lambda settings: fake_embedder)

        corpus.ensure_golden_corpus_indexed(eval_settings, corpus_dir=tiny_corpus_dir)
        vector_store = make_vector_store(eval_settings, fake_embedder)
        count_after_first_run = vector_store.count()
        fake_embedder.embed.reset_mock()

        corpus.ensure_golden_corpus_indexed(eval_settings, corpus_dir=tiny_corpus_dir)

        assert vector_store.count() == count_after_first_run
        fake_embedder.embed.assert_not_called()

    def test_defaults_to_the_real_golden_corpus_dir(
        self, eval_settings, fake_embedder, monkeypatch
    ):
        from src.evaluation import corpus
        from src.retrieval.vector_store import make_vector_store

        monkeypatch.setattr(corpus, "make_embedder", lambda settings: fake_embedder)

        corpus.ensure_golden_corpus_indexed(eval_settings)

        vector_store = make_vector_store(eval_settings, fake_embedder)
        assert vector_store.count() > 0

    def test_never_touches_production_chroma_persist_dir(
        self, eval_settings, fake_embedder, tiny_corpus_dir, monkeypatch
    ):
        from pathlib import Path

        from src.evaluation import corpus

        monkeypatch.setattr(corpus, "make_embedder", lambda settings: fake_embedder)

        corpus.ensure_golden_corpus_indexed(eval_settings, corpus_dir=tiny_corpus_dir)

        assert Path("./data/chroma") != eval_settings.chroma_persist_dir
        assert eval_settings.chroma_persist_dir == eval_settings.eval_chroma_persist_dir
