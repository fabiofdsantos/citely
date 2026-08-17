from __future__ import annotations

import pytest

from citely.ingest.chunking import chunk_document, chunk_text
from citely.models import RawDocument


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("A short paragraph.", chunk_size=100, overlap=10) == ["A short paragraph."]


def test_every_chunk_respects_the_size_limit() -> None:
    text = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(40))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 1


def test_paragraph_boundaries_are_preserved() -> None:
    """Chunks must not start or end mid-sentence when a boundary was available."""
    text = "\n\n".join([f"Sentence {i} ends here." for i in range(20)])
    for chunk in chunk_text(text, chunk_size=120, overlap=0):
        assert chunk.startswith("Sentence")
        assert chunk.endswith("here.")


def test_no_content_is_lost_without_overlap() -> None:
    paragraphs = [f"Paragraph {i}." for i in range(30)]
    chunks = chunk_text("\n\n".join(paragraphs), chunk_size=100, overlap=0)
    recovered = "\n".join(chunks)
    assert all(p in recovered for p in paragraphs)


def test_overlap_repeats_context_across_the_boundary() -> None:
    text = "\n\n".join(f"Paragraph {i} of the act." for i in range(20))
    with_overlap = chunk_text(text, chunk_size=150, overlap=60)
    without_overlap = chunk_text(text, chunk_size=150, overlap=0)
    assert len(with_overlap) > len(without_overlap)


def test_oversized_paragraph_is_split_at_sentences() -> None:
    paragraph = " ".join(f"Sentence number {i} is here." for i in range(50))
    chunks = chunk_text(paragraph, chunk_size=120, overlap=0)
    assert all(len(c) <= 120 for c in chunks)
    assert "Sentence number 0 is here." in chunks[0]


def test_unbreakable_text_is_hard_split() -> None:
    """A table row or long URL has no boundary; the size limit still holds."""
    chunks = chunk_text("x" * 500, chunk_size=100, overlap=0)
    assert [len(c) for c in chunks] == [100] * 5


def test_overlap_larger_than_chunk_size_is_rejected() -> None:
    """Otherwise each chunk is mostly the previous one and ingestion never ends."""
    with pytest.raises(ValueError, match="must be smaller"):
        chunk_text("text", chunk_size=100, overlap=100)


def test_whitespace_only_text_yields_nothing() -> None:
    assert chunk_text("   \n\n  \n ", chunk_size=100, overlap=0) == []


class TestChunkDocument:
    def test_provenance_is_carried_onto_every_chunk(self) -> None:
        document = RawDocument(
            source_uri="data/corpus/ai_act.txt",
            text="\n\n".join(f"Paragraph {i}." for i in range(20)),
            title="EU AI Act",
            metadata={"suffix": ".txt"},
        )

        chunks = chunk_document(document, chunk_size=100, overlap=0)

        assert len(chunks) > 1
        assert all(c.source_uri == "data/corpus/ai_act.txt" for c in chunks)
        assert all(c.title == "EU AI Act" for c in chunks)
        assert all(c.document_id == document.document_id for c in chunks)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_chunking_is_deterministic(self) -> None:
        """Same input, same ids — the precondition for incremental ingestion."""
        document = RawDocument(source_uri="s", text="\n\n".join(f"P{i}." for i in range(30)))

        first = chunk_document(document, chunk_size=80, overlap=10)
        second = chunk_document(document, chunk_size=80, overlap=10)

        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
