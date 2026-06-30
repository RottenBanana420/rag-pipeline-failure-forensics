from dataclasses import dataclass


@dataclass(frozen=True)
class VectorStoreHit:
    chunk_id: str
    text: str
    doc_id: str
    source_path: str
    title: str
    section_heading: str | None
    chunk_index: int
    strategy: str
    similarity: float
