"""Grounded generation prompt builder.

Turns a user question plus retrieved chunks (`VectorStoreHit`) into an
LLM-ready system/user prompt pair. Provider-agnostic — no LLM API calls are
made here; the resulting `GroundedPrompt` drops directly into either OpenAI's
or Anthropic's message format later.
"""

from dataclasses import dataclass

from src.retrieval.models import VectorStoreHit

GROUNDED_SYSTEM_PROMPT = """You are a grounded question-answering assistant.

Answer the user's question using only the information contained in the
numbered context blocks provided below. Never use outside knowledge, even if
you believe you know the answer.

Cite every claim inline with the bracket number(s) of the context block(s) it
came from, e.g. [1] for a claim drawn from a single block, or [1][3] for a
claim drawn from multiple blocks. Do not fabricate citations — every bracket
number you use must correspond to a context block that was actually used to
support that claim.

If the context blocks do not contain enough information to answer the
question, fully or partially, say so explicitly instead of guessing. Use
exactly this fallback phrase: "I don't have enough information in the provided context to answer this."
"""

NO_CONTEXT_MARKER = "(no context retrieved)"


@dataclass(frozen=True)
class GroundedPrompt:
    system: str
    user: str


def format_context_blocks(hits: list[VectorStoreHit]) -> str:
    """Render hits as numbered context blocks, 1-indexed in list order.

    The pipeline has already ranked `hits`; this function must not re-sort
    them. An empty list returns an explicit marker rather than an empty
    string, so the user prompt is never silently blank.
    """
    if not hits:
        return NO_CONTEXT_MARKER

    blocks = []
    for index, hit in enumerate(hits, start=1):
        attribution = hit.title
        if hit.section_heading:
            attribution += f" — {hit.section_heading}"
        blocks.append(f"[{index}] Source: {attribution}\n{hit.text}")

    return "\n\n".join(blocks)


def build_grounded_prompt(query: str, hits: list[VectorStoreHit]) -> GroundedPrompt:
    """Combine the system prompt with an XML-wrapped question and context."""
    user = (
        f"<context>\n{format_context_blocks(hits)}\n</context>\n\n"
        f"<question>\n{query}\n</question>"
    )
    return GroundedPrompt(system=GROUNDED_SYSTEM_PROMPT, user=user)
