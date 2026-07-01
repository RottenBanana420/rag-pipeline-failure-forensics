"""Vector store module — protocol, ChromaVectorStore implementation, and factory.

``VectorStoreProtocol`` defines the structural interface every vector store must satisfy.
``ChromaVectorStore`` is the ChromaDB-backed implementation.
``make_vector_store`` is a factory that reads ``settings.vector_store_provider`` and
returns the appropriate provider instance.
``VectorStore`` is kept as a backward-compatibility alias via module-level ``__getattr__``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import chromadb

from src.ingestion import Chunk
from src.retrieval.embedder import EmbedderProtocol
from src.retrieval.models import VectorStoreHit

if TYPE_CHECKING:
    from src.config import Settings

COLLECTION_NAME = "rag_chunks"

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Structural interface that every vector store provider must satisfy."""

    def filter_duplicates(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> tuple[list[Chunk], list[list[float]]]:
        """Return only chunks that are not near-duplicates of stored content."""
        ...

    def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> list[str]:
        """Insert or update chunks and their embeddings; return the stored IDs."""
        ...

    def query(self, embedding: list[float], k: int = 10) -> list[VectorStoreHit]:
        """Return the *k* nearest neighbours for a query embedding."""
        ...

    def get_by_ids(self, ids: list[str]) -> list[VectorStoreHit]:
        """Fetch specific chunks by their IDs."""
        ...

    def count(self) -> int:
        """Return the total number of stored chunks."""
        ...


class ChromaVectorStore:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        self._threshold = settings.dedup_threshold
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir_str)

        existing_names = [c.name for c in client.list_collections()]
        is_new = COLLECTION_NAME not in existing_names

        if is_new and embedder is not None:
            self._collection = client.get_or_create_collection(
                COLLECTION_NAME,
                metadata={
                    "embedding_provider": embedder.provider_id,
                    "embedding_dimensions": embedder.dimensions,
                },
                configuration={"hnsw": {"space": "cosine"}},
            )
        else:
            self._collection = client.get_or_create_collection(
                COLLECTION_NAME,
                configuration={"hnsw": {"space": "cosine"}},
            )

        if not is_new and embedder is not None:
            stored_meta = self._collection.metadata or {}
            stored_provider = stored_meta.get("embedding_provider")
            stored_dims = stored_meta.get("embedding_dimensions")
            if stored_provider is not None and stored_provider != embedder.provider_id:
                raise ValueError(
                    f"Collection was indexed with '{stored_provider}' ({stored_dims} dims). "
                    f"Current config is '{embedder.provider_id}' ({embedder.dimensions} dims). "
                    "Delete 'data/chroma/' and re-index to switch providers."
                )

    def filter_duplicates(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> tuple[list[Chunk], list[list[float]]]:
        if self._collection.count() == 0:
            return chunks, embeddings
        results = self._collection.query(
            query_embeddings=embeddings,  # type: ignore[arg-type]
            n_results=1,
            include=["distances"],
        )
        all_distances: list[list[float]] = results["distances"]  # type: ignore[assignment]
        accepted_chunks: list[Chunk] = []
        accepted_embeddings: list[list[float]] = []
        for chunk, embedding, distances in zip(
            chunks, embeddings, all_distances, strict=True
        ):
            is_duplicate = bool(distances) and (1.0 - distances[0]) >= self._threshold
            if is_duplicate:
                logger.debug(
                    "Skipping duplicate chunk %s (similarity=%.4f >= threshold=%.2f)",
                    chunk.chunk_id,
                    1.0 - distances[0],
                    self._threshold,
                )
            else:
                accepted_chunks.append(chunk)
                accepted_embeddings.append(embedding)
        skipped = len(chunks) - len(accepted_chunks)
        if skipped:
            logger.info(
                "Dedup: skipped %d/%d chunks as near-duplicates (threshold=%.2f)",
                skipped,
                len(chunks),
                self._threshold,
            )
        return accepted_chunks, accepted_embeddings

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> list[str]:
        if not chunks:
            return []
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_path": c.source_path,
                    "chunk_index": c.chunk_index,
                    "section_heading": c.section_heading or "",
                    "strategy": c.strategy,
                    "char_count": len(c.text),
                    "doc_id": c.doc_id,
                    "title": c.title,
                }
                for c in chunks
            ],
        )
        return [c.chunk_id for c in chunks]

    def query(self, embedding: list[float], k: int = 10) -> list[VectorStoreHit]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[VectorStoreHit] = []
        for chunk_id, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],  # type: ignore[index]
            results["metadatas"][0],  # type: ignore[index]
            results["distances"][0],  # type: ignore[index]
            strict=False,
        ):
            hits.append(
                VectorStoreHit(
                    chunk_id=chunk_id,
                    text=text,
                    doc_id=meta["doc_id"],
                    source_path=meta["source_path"],
                    title=meta["title"],
                    section_heading=meta["section_heading"] or None,
                    chunk_index=int(meta["chunk_index"]),  # pyright: ignore[reportArgumentType]
                    strategy=meta["strategy"],
                    similarity=1.0 - dist,
                )
            )
        return hits

    def get_by_ids(self, ids: list[str]) -> list[VectorStoreHit]:
        if not ids:
            return []
        results = self._collection.get(ids=ids, include=["documents", "metadatas"])
        hits: list[VectorStoreHit] = []
        for chunk_id, text, meta in zip(
            results["ids"],
            results["documents"],  # type: ignore[arg-type]
            results["metadatas"],  # type: ignore[arg-type]
            strict=True,
        ):
            hits.append(
                VectorStoreHit(
                    chunk_id=chunk_id,
                    text=text,
                    doc_id=meta["doc_id"],  # type: ignore[arg-type]
                    source_path=meta["source_path"],  # type: ignore[arg-type]
                    title=meta["title"],  # type: ignore[arg-type]
                    section_heading=meta["section_heading"] or None,  # type: ignore[arg-type]
                    chunk_index=int(meta["chunk_index"]),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
                    strategy=meta["strategy"],  # type: ignore[arg-type]
                    similarity=0.0,
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()


def make_vector_store(settings: Settings) -> VectorStoreProtocol:
    """Return a vector store instance for the provider specified in *settings*.

    Raises:
        NotImplementedError: If ``settings.vector_store_provider`` is ``"qdrant"``
            (not yet implemented).
        ValueError: If ``settings.vector_store_provider`` is not a recognised value.
    """
    provider = settings.vector_store_provider

    if provider == "chroma":
        return ChromaVectorStore(settings)

    if provider == "qdrant":
        raise NotImplementedError("qdrant vector store is not yet implemented")

    valid = "chroma, qdrant"
    raise ValueError(
        f"Unknown vector store provider: {provider!r}. Valid providers are: {valid}"
    )


def __getattr__(name: str) -> object:
    """Lazy loader for backward-compatibility aliases.

    Provides ``VectorStore`` as an alias for ``ChromaVectorStore`` without
    polluting the module namespace at import time.
    """
    if name == "VectorStore":
        globals()["VectorStore"] = ChromaVectorStore
        return ChromaVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
