"""Citation parser for extracting bracket-marked citations from LLM-generated answers.

Parses [N]-style citation markers (e.g., [1], [1][3]) from a text string and returns
a list of Citation objects, each pairing the text preceding the citation run with the
chunk indices referenced by that run.

This is a v1 heuristic implementation using regex to find citation markers. It does
not attempt sentence-boundary NLP — text is bounded by the positions of citation runs.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    """A claim and the chunk indices that support it.

    Attributes:
        claim_text: The text (claim) that preceded this citation run, stripped of
                   leading/trailing whitespace.
        chunk_indices: An ordered list of integer chunk IDs referenced by the citation
                      markers (e.g., [1][3] → [1, 3]).
    """

    claim_text: str
    chunk_indices: list[int]


def parse_citations(answer_text: str) -> list[Citation]:
    """Extract citations from an answer string.

    Finds all contiguous runs of bracket citation markers matching the pattern
    [N] (where N is one or more digits), and for each run, extracts:
    - claim_text: the text from the end of the previous run (or the start of the
                  string) to the start of this run, stripped.
    - chunk_indices: the ordered list of integers extracted from all brackets in
                     this run.

    Args:
        answer_text: The raw text string from an LLM response.

    Returns:
        A list of Citation objects, one per citation run found. Empty list if no
        citation markers are present.
    """
    pattern = r"(?:\[\d+\])+"
    matches = list(re.finditer(pattern, answer_text))

    if not matches:
        return []

    citations = []
    prev_end = 0

    for match in matches:
        # Extract text between previous citation (or start) and this citation
        claim_text = answer_text[prev_end : match.start()].strip()

        # Extract all indices from this citation run
        # Pattern inside the run: find all [N] occurrences
        run_text = match.group()
        indices_matches = re.findall(r"\[(\d+)\]", run_text)
        chunk_indices = [int(idx) for idx in indices_matches]

        citations.append(Citation(claim_text=claim_text, chunk_indices=chunk_indices))

        prev_end = match.end()

    return citations
