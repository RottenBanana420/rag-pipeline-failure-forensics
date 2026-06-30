import logging

import chromadb

from src.config import Settings
from src.ingestion import Chunk

COLLECTION_NAME = "rag_chunks"

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self._threshold = settings.dedup_threshold
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir_str)
        self._collection = client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
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
        for chunk, embedding, distances in zip(chunks, embeddings, all_distances, strict=True):
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

    def count(self) -> int:
        return self._collection.count()
