"""Generation module — grounded prompt construction, citation parsing/verification,
and confidence scoring (Phase 2)."""

from src.generation.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    GroundedPrompt,
    build_grounded_prompt,
)

__all__ = [
    "GROUNDED_SYSTEM_PROMPT",
    "GroundedPrompt",
    "build_grounded_prompt",
]
