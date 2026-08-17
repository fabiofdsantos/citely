"""Loader and pipeline tests.

The headline behaviour under test is incremental re-ingestion: running twice
over an unchanged corpus must cost zero embedding calls, and editing one file
must only re-embed that file's changed chunks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from citely.errors import IngestionError
from citely.ingest.loaders import load_corpus, load_file
from citely.ingest.pipeline import ingest_path
from citely.stores.base import VectorStore
from tests.fakes import FakeEmbedder, InMemoryVectorStore

CHUNKING = {"chunk_size": 120, "chunk_overlap": 0}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "act.md").write_text(
        "# EU AI Act\n\n" + "\n\n".join(f"Article {i} says something." for i in range(10)),
        encoding="utf-8",
    )
    (root / "notes.txt").write_text(
        "\n\n".join(f"Note {i} about the act." for i in range(6)), encoding="utf-8"
    )
    return root


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


class TestLoaders:
    def test_loads_supported_files_sorted(self, corpus: Path) -> None:
        documents = list(load_corpus(corpus))
        assert [Path(d.source_uri).name for d in documents] == ["act.md", "notes.txt"]

    def test_markdown_heading_becomes_the_title(self, corpus: Path) -> None:
        assert load_file(corpus / "act.md").title == "EU AI Act"

    def test_filename_is_the_fallback_title(self, corpus: Path) -> None:
        assert load_file(corpus / "notes.txt").title == "notes"

    def test_unsupported_files_are_ignored(self, corpus: Path) -> None:
        (corpus / "image.png").write_bytes(b"\x89PNG\r\n")
        assert len(list(load_corpus(corpus))) == 2

    def test_missing_path_is_an_ingestion_error(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="does not exist"):
            list(load_corpus(tmp_path / "nope"))

    def test_empty_corpus_is_an_ingestion_error(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(IngestionError, match="no documents found"):
            list(load_corpus(tmp_path / "empty"))

    def test_empty_file_is_an_ingestion_error(self, tmp_path: Path) -> None:
        blank = tmp_path / "blank.txt"
        blank.write_text("   \n", encoding="utf-8")
        with pytest.raises(IngestionError, match="is empty"):
            load_file(blank)

    def test_invalid_utf8_is_an_ingestion_error(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.txt"
        broken.write_bytes(b"\xff\xfe not utf-8")
        with pytest.raises(IngestionError, match="not valid UTF-8"):
            load_file(broken)

    def test_a_single_file_can_be_ingested_directly(self, corpus: Path) -> None:
        assert len(list(load_corpus(corpus / "act.md"))) == 1


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


class TestIngestion:
    async def test_first_run_embeds_everything(self, corpus: Path) -> None:
        embedder, store = FakeEmbedder(), InMemoryVectorStore()

        report = await ingest_path(corpus, embedder=embedder, store=store, **CHUNKING)

        assert report.embedded == len(store.chunks)
        assert report.skipped == 0
        assert not report.unchanged

    async def test_reingesting_unchanged_corpus_embeds_nothing(self, corpus: Path) -> None:
        """The idempotency guarantee, asserted on the paid operation."""
        embedder, store = FakeEmbedder(), InMemoryVectorStore()
        first = await ingest_path(corpus, embedder=embedder, store=store, **CHUNKING)
        calls_after_first = embedder.calls
        stored_after_first = dict(store.chunks)

        second = await ingest_path(corpus, embedder=embedder, store=store, **CHUNKING)

        assert embedder.calls == calls_after_first, "re-ingestion re-embedded unchanged chunks"
        assert second.embedded == 0
        assert second.skipped == first.embedded
        assert second.unchanged
        assert store.chunks.keys() == stored_after_first.keys()

    async def test_editing_one_paragraph_re_embeds_only_that_chunk(self, corpus: Path) -> None:
        embedder, store = FakeEmbedder(), InMemoryVectorStore()
        await ingest_path(corpus, embedder=embedder, store=store, **CHUNKING)
        embedder.embedded_texts.clear()

        notes = corpus / "notes.txt"
        notes.write_text(
            notes.read_text(encoding="utf-8").replace("Note 5", "Note 5 (amended)"),
            encoding="utf-8",
        )
        report = await ingest_path(corpus, embedder=embedder, store=store, **CHUNKING)

        assert report.embedded == 1
        assert report.deleted == 1
        assert "amended" in " ".join(embedder.embedded_texts)

    async def test_deleted_content_is_removed_from_the_store(self, corpus: Path) -> None:
        embedder, store = FakeEmbedder(), InMemoryVectorStore()
        await ingest_path(corpus, embedder=embedder, store=store, **CHUNKING)
        before = len(store.chunks)

        (corpus / "notes.txt").write_text("Only one note remains.", encoding="utf-8")
        report = await ingest_path(corpus, embedder=embedder, store=store, **CHUNKING)

        assert report.deleted > 0
        assert len(store.chunks) < before

    async def test_stored_chunks_keep_their_provenance(self, corpus: Path) -> None:
        embedder, store = FakeEmbedder(), InMemoryVectorStore()
        await ingest_path(corpus, embedder=embedder, store=store, **CHUNKING)

        stored = next(iter(store.chunks.values()))
        assert stored.chunk.source_uri.endswith((".md", ".txt"))
        assert stored.embedding_model == "fake-embed"

    async def test_report_is_per_document(self, corpus: Path) -> None:
        embedder, store = FakeEmbedder(), InMemoryVectorStore()

        report = await ingest_path(corpus, embedder=embedder, store=store, **CHUNKING)

        assert len(report.documents) == 2
        assert all(d.total_chunks > 0 for d in report.documents)


def test_in_memory_store_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryVectorStore(), VectorStore)
