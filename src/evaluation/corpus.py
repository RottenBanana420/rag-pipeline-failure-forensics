"""Idempotent indexing of the golden corpus into its own isolated vector store.

The golden corpus (`data/golden/corpus/*.md`) has no ingestion entrypoint of
its own yet (`scripts/seed_corpus.py` is still a Phase 1 stub) — the eval
runner indexes it directly via the same primitives `Indexer`/`Chunker`/
`DocumentLoader` already provide, always into `settings.chroma_persist_dir`
as passed in (callers pass a settings snapshot pointed at
`eval_chroma_persist_dir`, never production `./data/chroma`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.evaluation.dataset import CORPUS_DIR
from src.ingestion.chunker import Chunker
from src.ingestion.loader import DocumentLoader
from src.retrieval.embedder import make_embedder
from src.retrieval.indexer import Indexer
from src.retrieval.vector_store import make_vector_store

if TYPE_CHECKING:
    from src.config import Settings


def ensure_golden_corpus_indexed(
    settings: Settings, corpus_dir: Path = CORPUS_DIR
) -> None:
    """Index *corpus_dir* into `settings.chroma_persist_dir` unless already populated."""
    embedder = make_embedder(settings)
    vector_store = make_vector_store(settings, embedder)
    if vector_store.count() > 0:
        return

    loader = DocumentLoader(settings)
    chunker = Chunker(settings)
    indexer = Indexer(settings, embedder=embedder, vector_store=vector_store)

    docs = [
        doc for path in sorted(corpus_dir.glob("*.md")) for doc in loader.load(path)
    ]
    indexer.index(chunker.chunk(docs))
