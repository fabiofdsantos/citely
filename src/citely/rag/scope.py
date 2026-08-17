"""Scope checking: is the question even about what the corpus covers?

Citation verification proves a quote came from the corpus. It cannot prove the
quote answers the question — a measured failure of this system was returning EU
requirements, correctly cited, to a question about the *UK* AI Act. Every claim
was grounded; none of them was relevant. Verification is blind to that by
construction, so the check has to happen before generation.

The heuristic: pull the *identifying* words out of the question — proper nouns,
acronyms, numbered references — and require each to appear somewhere in the
retrieved sources. A question naming something the corpus never mentions cannot
be answered from that corpus, whatever the similarity scores say. Nothing here
is domain-specific: it works the same over legislation, clinical guidelines or
internal runbooks.

Deliberately conservative, because the failure being fixed is rarer than the
failure that a clumsy fix would introduce. Only tokens that appear *nowhere* in
the retrieved text count against the question, so "the European AI Regulation"
passes on the strength of its individual words even when that exact phrase is
absent.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from citely.models import ScoredChunk

#: A capitalised word, an all-caps acronym, or an ordinal reference such as
#: "Article 17" where the number carries the identity.
_CAPITALISED = re.compile(r"\b[A-Z][a-zA-Z]+\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
_NUMBERED_REFERENCE = re.compile(r"\b[A-Z][a-z]+\s+(\d{1,3})\b")

#: Words capitalised for grammar rather than identity. English function and
#: question words only — deliberately no domain vocabulary, so the check works
#: the same over case law, clinical guidelines or internal runbooks. Terms
#: specific to a corpus belong in ``CITELY_SCOPE_IGNORE_TERMS``.
_NOT_IDENTIFYING = frozenset(
    [
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "may",
        "must",
        "my",
        "not",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "there",
        "these",
        "they",
        "this",
        "to",
        "under",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


def identifying_terms(question: str, ignore: Collection[str] = ()) -> dict[str, str]:
    """Extract the words that determine which corpus could answer a question.

    Returns a mapping of comparison key (casefolded) to the surface form as the
    user wrote it, so a refusal can say "UK" rather than "uk".

    Args:
        question: The user's question.
        ignore: Extra terms to treat as non-identifying. A corpus has its own
            ubiquitous nouns — "Regulation" for legislation, "Patient" for
            clinical notes — which say nothing about whether the corpus can
            answer, and would otherwise be checked as if they did.
    """
    skip = _NOT_IDENTIFYING | {term.casefold() for term in ignore}
    terms: dict[str, str] = {}

    for match in _ACRONYM.finditer(question):
        term = match.group()
        if term.casefold() not in skip:
            terms[term.casefold()] = term

    for match in _CAPITALISED.finditer(question):
        term = match.group()
        # Skip the first word of the question: it is capitalised by grammar, so
        # its capitalisation carries no signal about identity.
        if match.start() == 0:
            continue
        if term.casefold() not in skip:
            terms.setdefault(term.casefold(), term)

    # "Article 17" is a different thing from "Article 5"; the number is the
    # identity. Bare numbers are ignored — "the 3 obligations" identifies
    # nothing and would cause false refusals.
    for match in _NUMBERED_REFERENCE.finditer(question):
        number = match.group(1)
        terms.setdefault(number, f"{match.group().split()[0]} {number}")

    return terms


def out_of_scope_terms(
    question: str, sources: Sequence[ScoredChunk], ignore: Collection[str] = ()
) -> set[str]:
    """Return the question's identifying terms that no source mentions.

    Terms are returned in the surface form the user wrote them in, for use in
    the refusal message.
    """
    if not sources:
        return set()

    haystack = " ".join(scored.chunk.text for scored in sources).casefold()
    # Word-boundary matching, so "uk" does not match inside "Denmark".
    return {
        surface
        for key, surface in identifying_terms(question, ignore).items()
        if not re.search(rf"\b{re.escape(key)}\b", haystack)
    }


def scope_refusal_reason(
    question: str, sources: Sequence[ScoredChunk], ignore: Collection[str] = ()
) -> str | None:
    """Return a refusal reason if the question is out of scope, else None."""
    missing = out_of_scope_terms(question, sources, ignore)
    if not missing:
        return None

    named = ", ".join(sorted(missing))
    return (
        f"The corpus does not mention {named}, so it cannot answer a question "
        "about that. The retrieved passages are about a different subject."
    )
