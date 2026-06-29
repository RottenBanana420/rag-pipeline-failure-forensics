from __future__ import annotations

import re
from datetime import UTC, datetime

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings
from src.ingestion.models import Chunk, ChunkingStrategy, ProcessedDocument, chunk_id


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Chunker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def chunk(self, docs: list[ProcessedDocument]) -> list[Chunk]:
        strategy = self._settings.chunk_strategy
        if strategy == "fixed_size":
            return self._fixed_size(docs)
        if strategy == "recursive_header":
            return self._recursive_header(docs)
        if strategy == "semantic":
            return self._semantic(docs)
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _fixed_size(self, docs: list[ProcessedDocument]) -> list[Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        return self._apply_splitter(docs, splitter, "fixed_size")

    def _recursive_header(self, docs: list[ProcessedDocument]) -> list[Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        return self._apply_splitter(docs, splitter, "recursive_header")

    def _semantic(self, docs: list[ProcessedDocument]) -> list[Chunk]:
        from openai import OpenAI

        client = OpenAI(api_key=self._settings.openai_api_key)
        chunks: list[Chunk] = []
        now = _now_iso()
        chunk_idx = 0

        for doc in docs:
            if not doc.text.strip():
                continue

            sentences = _sentence_split(doc.text)
            if len(sentences) <= 1:
                chunks.append(self._make_chunk(doc, doc.text.strip(), chunk_idx, "semantic", now))
                chunk_idx += 1
                continue

            resp = client.embeddings.create(
                model=self._settings.embedding_model,
                input=sentences,
            )
            vecs = np.array([e.embedding for e in resp.data], dtype=float)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            normalized = vecs / np.maximum(norms, 1e-8)
            cosine_sims = (normalized[:-1] * normalized[1:]).sum(axis=1)
            distances = 1.0 - cosine_sims

            threshold = float(
                np.percentile(distances, self._settings.semantic_breakpoint_percentile)
            )
            breakpoint_set = {int(i) for i in np.where(distances > threshold)[0]}

            group: list[str] = [sentences[0]]
            for i, sentence in enumerate(sentences[1:], start=1):
                if (i - 1) in breakpoint_set:
                    text = " ".join(group).strip()
                    if text:
                        chunks.append(self._make_chunk(doc, text, chunk_idx, "semantic", now))
                        chunk_idx += 1
                    group = [sentence]
                else:
                    group.append(sentence)

            text = " ".join(group).strip()
            if text:
                chunks.append(self._make_chunk(doc, text, chunk_idx, "semantic", now))
                chunk_idx += 1

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_splitter(
        self,
        docs: list[ProcessedDocument],
        splitter: RecursiveCharacterTextSplitter,
        strategy: ChunkingStrategy,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        now = _now_iso()
        chunk_idx = 0
        for doc in docs:
            if not doc.text.strip():
                continue
            for text in splitter.split_text(doc.text):
                if text.strip():
                    chunks.append(self._make_chunk(doc, text.strip(), chunk_idx, strategy, now))
                    chunk_idx += 1
        return chunks

    def _make_chunk(
        self,
        doc: ProcessedDocument,
        text: str,
        chunk_index: int,
        strategy: ChunkingStrategy,
        processed_at: str,
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id(doc.doc_id, text),
            doc_id=doc.doc_id,
            source_path=doc.source_path,
            source_format=doc.source_format,
            title=doc.title,
            section_heading=doc.section_heading,
            page_number=doc.page_number,
            text=text,
            chunk_index=chunk_index,
            strategy=strategy,
            processed_at=processed_at,
        )


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s.strip()]
