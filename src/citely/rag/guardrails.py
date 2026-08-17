"""Guardrails: input validation and citation verification.

The verification step is what separates this service from a chatbot with a
retrieval step bolted on. The model claims a quote supports a claim; we check
that the quote actually appears in the chunk it names. Unverifiable citations
are dropped, and an answer left with no verified citation is converted into a
refusal rather than shown.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from citely.models import Citation, ScoredChunk

#: Long enough that a query carries intent, short enough to bound prompt size
#: and cost. Rejecting oversized input is also the cheapest defence against
#: someone pasting an entire instruction payload into the question field.
MAX_QUERY_LENGTH = 1000
MIN_QUERY_LENGTH = 3

#: A quote shorter than this verifies trivially against almost any text ("the",
#: "AI"), so it proves nothing about grounding.
MIN_QUOTE_LENGTH = 12

_WHITESPACE = re.compile(r"\s+")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InvalidQueryError(ValueError):
    """The question cannot be processed as written."""


def validate_query(query: str) -> str:
    """Normalise and bounds-check a user question.

    Raises:
        InvalidQueryError: if the query is empty, too long, or malformed.
    """
    # NFKC first: without it, visually identical look-alike characters slip
    # past every length and content check that follows.
    cleaned = unicodedata.normalize("NFKC", query)
    cleaned = _CONTROL_CHARS.sub("", cleaned).strip()

    if len(cleaned) < MIN_QUERY_LENGTH:
        raise InvalidQueryError("the question is empty or too short")
    if len(cleaned) > MAX_QUERY_LENGTH:
        raise InvalidQueryError(
            f"the question is {len(cleaned)} characters; the limit is {MAX_QUERY_LENGTH}"
        )
    return cleaned


def _normalise(text: str) -> str:
    """Collapse whitespace so quotes survive re-wrapping and indentation."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def quote_is_grounded(quote: str, chunk_text: str) -> bool:
    """Whether a quote genuinely appears in the text it claims to come from."""
    if len(quote.strip()) < MIN_QUOTE_LENGTH:
        return False
    return _normalise(quote) in _normalise(chunk_text)


def verify_citations(
    claimed: Sequence[tuple[int, str]],
    sources: Sequence[ScoredChunk],
) -> tuple[list[Citation], list[str]]:
    """Check each claimed ``(source_number, quote)`` against the retrieved text.

    Returns the verified citations and a list of human-readable rejection
    reasons, which are logged rather than shown: they describe model failures,
    not user errors.
    """
    verified: list[Citation] = []
    rejected: list[str] = []
    seen: set[tuple[str, str]] = set()

    for number, quote in claimed:
        if not 1 <= number <= len(sources):
            rejected.append(f"source {number} was never retrieved")
            continue

        chunk = sources[number - 1].chunk
        if not quote_is_grounded(quote, chunk.text):
            rejected.append(f"quote for source {number} does not appear in that source")
            continue

        key = (chunk.chunk_id, _normalise(quote))
        if key in seen:
            continue
        seen.add(key)

        verified.append(
            Citation(
                chunk_id=chunk.chunk_id,
                quote=quote.strip(),
                source_uri=chunk.source_uri,
                title=chunk.title,
            )
        )

    return verified, rejected
