"""Citation parser for extracting bracket-marked citations from LLM-generated answers.

Parses [N]-style citation markers (e.g., [1], [1][3]) from a text string and returns
a list of Citation objects, each pairing the claim text a citation run supports with
the chunk indices referenced by that run.

This is a v1 heuristic implementation using regex to find citation markers. It does
not attempt sentence-boundary NLP — text is bounded by the positions of citation runs
and, for leading markers (see below), the nearest sentence-terminal punctuation.

Markers are expected to follow the claim they support (`GROUNDED_SYSTEM_PROMPT`
instructs the model this way), so claim text is normally taken from the text
*preceding* a run. Models don't always comply — a run can open a sentence or
clause instead (e.g. "According to the context, [1] the rotation is weekly").
When the text preceding a run is empty or ends with a comma — the structural
signature of a lead-in with no real claim yet — the run is treated as a
*leading marker*: claim text is instead drawn by scanning forward to the
nearest sentence-terminal punctuation or the next citation run, and merged
with whatever (possibly empty) text preceded the marker.
"""

import re
from dataclasses import dataclass

_CITATION_RUN = re.compile(r"(?:\[\d+\])+")
_BRACKET_INDEX = re.compile(r"\[(\d+)\]")
_SENTENCE_TERMINATOR = re.compile(r"[.!?]")


@dataclass(frozen=True)
class Citation:
    """A claim and the chunk indices that support it.

    Attributes:
        claim_text: The claim text this citation run supports, stripped of
                   leading/trailing whitespace. Normally the text preceding the
                   run; for a leading marker (see module docstring), text found
                   by scanning forward is merged in instead.
        chunk_indices: An ordered list of integer chunk IDs referenced by the citation
                      markers (e.g., [1][3] → [1, 3]).
    """

    claim_text: str
    chunk_indices: list[int]


def parse_citations(answer_text: str) -> list[Citation]:
    """Extract citations from an answer string.

    Finds all contiguous runs of bracket citation markers matching the pattern
    [N] (where N is one or more digits). For each run:
    - chunk_indices: the ordered list of integers extracted from all brackets in
                     this run.
    - claim_text: the text from the end of the previous run (or the start of the
                  string) to the start of this run, stripped. If that text is
                  empty or ends with a comma (a leading marker — see module
                  docstring), text is instead scanned forward from the end of
                  this run to the nearer of the next sentence-terminal
                  punctuation (`.`, `!`, `?`) or the next citation run (end of
                  string if neither exists), and merged with the (possibly
                  empty) preceding text.

    Known limitation: if a leading marker's forward scan is bounded by another
    citation run with no sentence-terminal punctuation between them, that next
    run's own preceding text is left empty by the scan, so it may itself be
    treated as a leading marker even if the model intended it as a normal
    trailing citation. Accepted for a v1 heuristic — `GROUNDED_SYSTEM_PROMPT`
    instructs the model to keep multi-chunk citations for one claim as a single
    contiguous run (e.g. [1][3]), which is the case that would otherwise
    trigger this.

    Args:
        answer_text: The raw text string from an LLM response.

    Returns:
        A list of Citation objects, one per citation run found. Empty list if no
        citation markers are present.
    """
    matches = list(_CITATION_RUN.finditer(answer_text))

    if not matches:
        return []

    citations = []
    prev_end = 0

    for i, match in enumerate(matches):
        preceding = answer_text[prev_end : match.start()].strip()
        chunk_indices = [int(idx) for idx in _BRACKET_INDEX.findall(match.group())]

        if preceding == "" or preceding.endswith(","):
            next_match_start = (
                matches[i + 1].start() if i + 1 < len(matches) else len(answer_text)
            )
            forward_region = answer_text[match.end() : next_match_start]
            terminator = _SENTENCE_TERMINATOR.search(forward_region)
            boundary = (
                match.end() + terminator.end() if terminator else next_match_start
            )
            forward_text = answer_text[match.end() : boundary].strip()
            claim_text = (
                f"{preceding} {forward_text}".strip() if preceding else forward_text
            )
            prev_end = boundary
        else:
            claim_text = preceding
            prev_end = match.end()

        citations.append(Citation(claim_text=claim_text, chunk_indices=chunk_indices))

    return citations
