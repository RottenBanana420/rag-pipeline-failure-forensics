"""Generation module — grounded prompt construction, citation parsing/verification,
and confidence scoring (Phase 2)."""

from src.generation.citation_parser import Citation, parse_citations
from src.generation.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    INSUFFICIENT_CONTEXT_RESPONSE,
    GroundedPrompt,
    build_grounded_prompt,
    wrap_with_nonce,
)

__all__ = [
    "Citation",
    "GROUNDED_SYSTEM_PROMPT",
    "INSUFFICIENT_CONTEXT_RESPONSE",
    "GroundedPrompt",
    "build_grounded_prompt",
    "parse_citations",
    "wrap_with_nonce",
]
