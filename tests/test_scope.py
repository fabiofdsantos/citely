"""Scope-check tests.

The check exists to stop one specific measured failure: a grounded, correctly
cited answer to a question the corpus does not actually cover. The risk in
fixing it is trading that rare failure for a common one — refusing questions
that were perfectly answerable — so most of these tests are about *not* firing.
"""

from __future__ import annotations

import pytest

from citely.models import Chunk, ScoredChunk
from citely.rag.scope import identifying_terms, out_of_scope_terms, scope_refusal_reason

CORPUS = (
    "This Regulation establishes harmonised rules for placing artificial "
    "intelligence systems on the market in the European Union. Article 5 "
    "prohibits social scoring by public authorities. High-risk systems must "
    "undergo a conformity assessment."
)


def sources(text: str = CORPUS) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(document_id="d", text=text, index=0, source_uri="corpus/act.md"),
            score=0.9,
        )
    ]


class TestTermExtraction:
    def test_finds_proper_nouns(self) -> None:
        assert identifying_terms("Does the Brussels office apply?")["brussels"] == "Brussels"

    def test_finds_acronyms(self) -> None:
        assert identifying_terms("What does GDPR require?")["gdpr"] == "GDPR"

    def test_ignores_the_first_word(self) -> None:
        """Capitalised by grammar, so it carries no identity signal."""
        assert "what" not in identifying_terms("What does Article 5 prohibit?")

    def test_ignores_generic_domain_words(self) -> None:
        terms = identifying_terms("What obligations apply to AI Systems under the Regulation?")
        assert terms == {}

    def test_captures_article_numbers(self) -> None:
        assert "17" in identifying_terms("What does Article 17 say?")

    def test_ignores_bare_numbers(self) -> None:
        """ "the 3 obligations" identifies nothing and must not trigger a refusal."""
        assert identifying_terms("What are the 3 obligations?") == {}


class TestScopeDecision:
    def test_question_about_a_different_jurisdiction_is_out_of_scope(self) -> None:
        """The measured failure this check exists for."""
        assert scope_refusal_reason("What does the UK AI Act require?", sources()) is not None

    def test_question_about_another_regulation_is_out_of_scope(self) -> None:
        assert scope_refusal_reason("What does GDPR Article 17 say?", sources()) is not None

    def test_covered_question_passes(self) -> None:
        assert scope_refusal_reason("What does Article 5 prohibit?", sources()) is None

    def test_question_with_no_proper_nouns_passes(self) -> None:
        assert scope_refusal_reason("who must run a conformity assessment?", sources()) is None

    def test_paraphrased_entity_passes_on_its_parts(self) -> None:
        """ "European AI Regulation" is absent verbatim but present word by word."""
        assert scope_refusal_reason("What does the European Regulation cover?", sources()) is None

    def test_lowercase_question_is_not_penalised(self) -> None:
        """Typed casually, with no capitals at all — must behave the same."""
        assert scope_refusal_reason("what does article 5 prohibit?", sources()) is None

    def test_empty_retrieval_defers_to_the_normal_refusal_path(self) -> None:
        assert scope_refusal_reason("What does the UK AI Act require?", []) is None

    def test_reason_names_the_missing_term(self) -> None:
        reason = scope_refusal_reason("What does the UK AI Act require?", sources())
        assert reason is not None
        assert "UK" in reason, "the refusal should echo the term as the user wrote it"

    def test_word_boundaries_are_respected(self) -> None:
        """ "uk" must not be considered present because "Denmark" contains it."""
        assert out_of_scope_terms("Does the UK apply?", sources("Denmark and Sweden")) == {"UK"}


@pytest.mark.parametrize(
    "question",
    [
        "What does Article 5 prohibit?",
        "Do high-risk systems need a conformity assessment?",
        "Who must establish a risk management system?",
        "what is prohibited for public authorities?",
        "Are providers of high-risk systems required to keep logs?",
    ],
)
def test_answerable_questions_are_never_refused(question: str) -> None:
    """The regression that matters: this must not create new over-refusals."""
    assert scope_refusal_reason(question, sources()) is None
