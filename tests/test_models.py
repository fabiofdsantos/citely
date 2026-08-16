import pytest
from pydantic import ValidationError

from citely.models import (
    Answer,
    Chunk,
    Citation,
    EmbeddedChunk,
    RawDocument,
    RetrievalResult,
    ScoredChunk,
    chunk_id_for,
)


def make_chunk(text: str = "Article 5 prohibits certain AI practices.", index: int = 0) -> Chunk:
    return Chunk(
        document_id="doc123",
        text=text,
        index=index,
        source_uri="data/corpus/ai_act.txt",
    )


class TestIdentity:
    def test_document_id_is_stable_for_the_same_source(self) -> None:
        doc = RawDocument(source_uri="data/corpus/ai_act.txt", text="hello")
        same = RawDocument(source_uri="data/corpus/ai_act.txt", text="different body")
        assert doc.document_id == same.document_id

    def test_chunk_id_is_content_addressed(self) -> None:
        """Same content, same id — this is what makes re-ingestion idempotent."""
        assert make_chunk().chunk_id == make_chunk().chunk_id

    def test_chunk_id_ignores_position(self) -> None:
        """Editing an early paragraph must not invalidate every later chunk."""
        assert make_chunk(index=0).chunk_id == make_chunk(index=7).chunk_id

    def test_chunk_id_changes_with_content(self) -> None:
        assert make_chunk(text="a").chunk_id != make_chunk(text="b").chunk_id

    def test_chunk_id_is_scoped_to_its_document(self) -> None:
        assert chunk_id_for("doc-a", "same text") != chunk_id_for("doc-b", "same text")


class TestValueSemantics:
    def test_chunks_are_immutable(self) -> None:
        chunk = make_chunk()
        with pytest.raises(ValidationError):
            chunk.text = "tampered"

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                document_id="d",
                text="t",
                index=0,
                source_uri="s",
                typo_field="oops",  # type: ignore[call-arg]
            )

    def test_embedded_chunk_reports_dimensions(self) -> None:
        embedded = EmbeddedChunk(
            chunk=make_chunk(),
            embedding=[0.1, 0.2, 0.3],
            embedding_model="nomic-embed-text",
        )
        assert embedded.dimensions == 3

    def test_scores_outside_the_unit_interval_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScoredChunk(chunk=make_chunk(), score=1.4)


class TestRetrievalResult:
    def test_empty_result_has_zero_top_score(self) -> None:
        result = RetrievalResult(query="what is a high-risk system?")
        assert result.is_empty
        assert result.top_score == 0.0

    def test_top_score_is_the_maximum(self) -> None:
        result = RetrievalResult(
            query="q",
            chunks=[
                ScoredChunk(chunk=make_chunk("a"), score=0.42),
                ScoredChunk(chunk=make_chunk("b"), score=0.91),
            ],
        )
        assert result.top_score == pytest.approx(0.91)


class TestAnswerInvariant:
    """An answer either refuses or cites. Nothing else is representable."""

    def test_grounded_answer_needs_citations(self) -> None:
        with pytest.raises(ValidationError, match="must cite at least one source"):
            Answer(query="q", text="Yes, definitely.")

    def test_refusal_needs_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="refusal_reason"):
            Answer(query="q", text="I can't answer that.", refused=True)

    def test_refusal_must_not_cite(self) -> None:
        with pytest.raises(ValidationError, match="must not carry citations"):
            Answer(
                query="q",
                text="Out of scope.",
                refused=True,
                refusal_reason="not covered by the corpus",
                citations=[Citation(chunk_id="c1", quote="q", source_uri="s")],
            )

    def test_valid_grounded_answer(self) -> None:
        answer = Answer(
            query="q",
            text="Article 5 prohibits certain practices [1].",
            citations=[
                Citation(
                    chunk_id="c1",
                    quote="Article 5 prohibits certain AI practices.",
                    source_uri="data/corpus/ai_act.txt",
                )
            ],
        )
        assert not answer.refused

    def test_valid_refusal(self) -> None:
        answer = Answer(
            query="q",
            text="The corpus does not cover this.",
            refused=True,
            refusal_reason="no chunk scored above the relevance threshold",
        )
        assert answer.refused
