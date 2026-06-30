"""Utility to generate a minimal valid 2-page PDF for testing.

Not a test file — called by conftest fixtures. Writes raw PDF bytes
so there is no runtime dependency on reportlab or fpdf2.
"""

from __future__ import annotations

from pathlib import Path


def create_sample_pdf(path: Path) -> None:
    """Write a deterministic 2-page PDF with extractable text to *path*."""
    page1_stream = b"BT /F1 12 Tf 72 720 Td (Page one content for testing.) Tj ET"
    page2_stream = b"BT /F1 12 Tf 72 720 Td (Page two content for testing.) Tj ET"

    body_parts: list[bytes] = [b"%PDF-1.4\n"]
    offsets: list[int] = []

    def record(content: bytes) -> None:
        offsets.append(sum(len(p) for p in body_parts))
        body_parts.append(content)

    record(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    record(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n")
    record(
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Contents 5 0 R /Resources << /Font << /F1 7 0 R >> >> >>\n"
        b"endobj\n"
    )
    record(
        b"4 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>\n"
        b"endobj\n"
    )
    record(
        f"5 0 obj\n<< /Length {len(page1_stream)} >>\nstream\n".encode()
        + page1_stream
        + b"\nendstream\nendobj\n"
    )
    record(
        f"6 0 obj\n<< /Length {len(page2_stream)} >>\nstream\n".encode()
        + page2_stream
        + b"\nendstream\nendobj\n"
    )
    record(b"7 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    xref_offset = sum(len(p) for p in body_parts)
    n_objects = len(offsets)

    xref = f"xref\n0 {n_objects + 1}\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"

    trailer = (
        f"trailer\n<< /Size {n_objects + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )

    path.write_bytes(b"".join(body_parts) + xref.encode() + trailer.encode())
