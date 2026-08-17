"""Structure-aware chunking.

Fixed-width slicing cuts sentences in half, which shows up later as citations
that quote fragments and as embeddings of text that means nothing on its own.
So text is first broken into natural segments — paragraphs, then sentences, then
(only if a single sentence is still oversized) hard slices — and chunks are
packed from whole segments.

Overlap is applied by re-including whole trailing segments rather than a
character window, so every chunk still starts at a real boundary.
"""

from __future__ import annotations

import re

from citely.models import Chunk, RawDocument

#: One or more blank lines separate paragraphs.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")

#: A sentence ends at ., ! or ? followed by whitespace. Deliberately naive: it
#: only ever runs on paragraphs already too big to fit, and the failure mode of
#: an over-eager split ("Art. 5") is a slightly small chunk, not a wrong one.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _hard_split(text: str, limit: int) -> list[str]:
    """Last resort for a segment with no internal boundary (tables, long URLs)."""
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _segments(text: str, limit: int) -> list[str]:
    """Break text into the largest natural units that each fit within ``limit``."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        stripped = paragraph.strip()
        if not stripped:
            continue
        if len(stripped) <= limit:
            units.append(stripped)
            continue

        for sentence in _SENTENCE_END.split(stripped):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= limit:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, limit))
    return units


def _overlap_tail(segments: list[str], overlap: int) -> list[str]:
    """Return the trailing segments that fit within the overlap budget.

    Carrying context across the boundary keeps a chunk answerable when the
    sentence that gives it meaning sat just above the cut.
    """
    if overlap <= 0:
        return []

    tail: list[str] = []
    budget = overlap
    for segment in reversed(segments):
        # Never let the carried context alone fill the next chunk: that would
        # make progress slow to a crawl and duplicate most of the corpus.
        if len(segment) > budget:
            break
        tail.insert(0, segment)
        budget -= len(segment) + 1
    return tail


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split raw text into overlapping, boundary-aligned chunks."""
    if overlap >= chunk_size:
        msg = f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        raise ValueError(msg)

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for segment in _segments(text, chunk_size):
        # +1 for the newline that will join this segment to the previous one.
        addition = len(segment) + (1 if current else 0)
        if current and length + addition > chunk_size:
            chunks.append("\n".join(current))
            current = _overlap_tail(current, overlap)
            length = sum(len(s) + 1 for s in current)
            addition = len(segment) + (1 if current else 0)

        current.append(segment)
        length += addition

    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_document(document: RawDocument, *, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk a loaded document, carrying its provenance onto every chunk."""
    return [
        Chunk(
            document_id=document.document_id,
            text=text,
            index=index,
            source_uri=document.source_uri,
            title=document.title,
            metadata=document.metadata,
        )
        for index, text in enumerate(
            chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        )
    ]
