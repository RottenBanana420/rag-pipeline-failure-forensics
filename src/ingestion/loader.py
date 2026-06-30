from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.config import Settings
from src.ingestion.models import ProcessedDocument

_SUPPORTED = {".md", ".txt", ".html", ".htm", ".pdf"}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _doc_id(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


class DocumentLoader:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def load(self, path: Path) -> list[ProcessedDocument]:
        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED:
            raise ValueError(
                f"Unsupported file extension: {suffix!r}. Supported: {_SUPPORTED}"
            )

        raw_bytes = path.read_bytes()
        if not raw_bytes.strip():
            return []

        source_path = str(path)
        did = _doc_id(raw_bytes)

        if suffix == ".md":
            return self._load_markdown(path, raw_bytes, source_path, did)
        if suffix == ".txt":
            return self._load_text(path, raw_bytes, source_path, did)
        if suffix in {".html", ".htm"}:
            return self._load_html(path, raw_bytes, source_path, did)
        # .pdf
        return self._load_pdf(path, source_path, did)

    # ------------------------------------------------------------------
    # Format-specific handlers
    # ------------------------------------------------------------------

    def _load_markdown(
        self, path: Path, raw_bytes: bytes, source_path: str, did: str
    ) -> list[ProcessedDocument]:
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        splits: list[tuple[int, str]] = [
            (m.start(), m.group(2).strip()) for m in heading_pattern.finditer(raw_text)
        ]

        if not splits:
            # No headings — whole file as one doc
            text = _strip_markdown(raw_text)
            return [
                ProcessedDocument(
                    doc_id=did,
                    source_path=source_path,
                    source_format="markdown",
                    title=path.stem,
                    section_heading=None,
                    page_number=None,
                    text=text.strip(),
                    processed_at=_now_iso(),
                )
            ]

        title = splits[0][1]
        docs: list[ProcessedDocument] = []
        now = _now_iso()

        # Intro section before first heading
        intro_text = _strip_markdown(raw_text[: splits[0][0]]).strip()
        if intro_text:
            docs.append(
                ProcessedDocument(
                    doc_id=did,
                    source_path=source_path,
                    source_format="markdown",
                    title=title,
                    section_heading=None,
                    page_number=None,
                    text=intro_text,
                    processed_at=now,
                )
            )

        for i, (start, heading) in enumerate(splits):
            end = splits[i + 1][0] if i + 1 < len(splits) else len(raw_text)
            section_raw = raw_text[start:end]
            # Remove the heading line itself
            section_body = re.sub(r"^#{1,6}\s+.+\n?", "", section_raw, count=1)
            text = _strip_markdown(section_body).strip()
            if not text:
                continue
            docs.append(
                ProcessedDocument(
                    doc_id=did,
                    source_path=source_path,
                    source_format="markdown",
                    title=title,
                    section_heading=heading,
                    page_number=None,
                    text=text,
                    processed_at=now,
                )
            )

        return docs

    def _load_text(
        self, path: Path, raw_bytes: bytes, source_path: str, did: str
    ) -> list[ProcessedDocument]:
        text = raw_bytes.decode("utf-8", errors="replace").strip()
        return [
            ProcessedDocument(
                doc_id=did,
                source_path=source_path,
                source_format="text",
                title=path.stem,
                section_heading=None,
                page_number=None,
                text=text,
                processed_at=_now_iso(),
            )
        ]

    def _load_html(
        self, path: Path, raw_bytes: bytes, source_path: str, did: str
    ) -> list[ProcessedDocument]:
        soup = BeautifulSoup(raw_bytes, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
        body = soup.body or soup

        title: str | None = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        nodes = list(body.children)
        sections: list[tuple[str | None, list[object]]] = []
        current_heading: str | None = None
        current_nodes: list[object] = []

        for node in nodes:
            tag_name = getattr(node, "name", None)
            if tag_name in heading_tags:
                sections.append((current_heading, current_nodes))
                current_heading = node.get_text(separator=" ", strip=True)
                current_nodes = []
                if title is None and tag_name == "h1":
                    title = current_heading
            else:
                current_nodes.append(node)
        sections.append((current_heading, current_nodes))

        if title is None:
            title = path.stem

        docs: list[ProcessedDocument] = []
        now = _now_iso()

        for heading, content_nodes in sections:
            text = " ".join(
                n.get_text(separator=" ", strip=True)
                for n in content_nodes
                if hasattr(n, "get_text")
            ).strip()
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            docs.append(
                ProcessedDocument(
                    doc_id=did,
                    source_path=source_path,
                    source_format="html",
                    title=title,
                    section_heading=heading,
                    page_number=None,
                    text=text,
                    processed_at=now,
                )
            )

        return docs

    def _load_pdf(
        self, path: Path, source_path: str, did: str
    ) -> list[ProcessedDocument]:
        reader = PdfReader(str(path))
        docs: list[ProcessedDocument] = []
        now = _now_iso()

        # Use filename stem as title; PDF metadata title if available
        title = path.stem
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title.strip() or title

        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            docs.append(
                ProcessedDocument(
                    doc_id=did,
                    source_path=source_path,
                    source_format="pdf",
                    title=title,
                    section_heading=None,
                    page_number=i,
                    text=text,
                    processed_at=now,
                )
            )

        return docs


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _strip_markdown(text: str) -> str:
    """Remove common Markdown syntax, leaving readable plaintext."""
    # Code fences
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", lambda m: m.group(0)[1:-1], text)
    # Links and images
    text = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)
    # Bold / italic
    text = re.sub(r"\*{1,3}([^\*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # List markers
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Blockquotes
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # HTML tags that may appear in Markdown
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
