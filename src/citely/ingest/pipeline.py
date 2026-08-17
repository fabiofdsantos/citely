"""Ingestion pipeline: load, chunk, embed, store.

The one interesting property here is incrementality. Chunk ids are content
hashes, so for each document we can compare the ids we just produced against the
ids already stored and act on the difference:

* ids in both sets     -> unchanged, skip (never re-embedded)
* ids only in the new  -> added, embed and write
* ids only in the old  -> the text they covered is gone, delete

Embedding is the expensive, paid step, so "skip" is the case worth optimising
for: re-running ingestion over an unchanged corpus costs zero embedding calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from citely.ingest.chunking import chunk_document
from citely.ingest.loaders import load_corpus
from citely.models import Chunk, EmbeddedChunk, RawDocument
from citely.providers.base import EmbeddingProvider
from citely.stores.base import VectorStore


class DocumentReport(BaseModel):
    """What ingestion did to a single document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_uri: str
    document_id: str
    total_chunks: int = Field(ge=0)
    embedded: int = Field(ge=0)
    skipped: int = Field(ge=0)
    deleted: int = Field(ge=0)

    @property
    def unchanged(self) -> bool:
        return self.embedded == 0 and self.deleted == 0


class IngestReport(BaseModel):
    """The outcome of one ingestion run, printable and assertable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: list[DocumentReport] = Field(default_factory=list)

    @property
    def embedded(self) -> int:
        return sum(d.embedded for d in self.documents)

    @property
    def skipped(self) -> int:
        return sum(d.skipped for d in self.documents)

    @property
    def deleted(self) -> int:
        return sum(d.deleted for d in self.documents)

    @property
    def unchanged(self) -> bool:
        """True when the run was a no-op, which is what a re-run should be."""
        return self.embedded == 0 and self.deleted == 0


async def ingest_document(
    document: RawDocument,
    *,
    embedder: EmbeddingProvider,
    store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> DocumentReport:
    """Bring one document's stored chunks in line with its current content."""
    chunks = chunk_document(document, chunk_size=chunk_size, overlap=chunk_overlap)

    # Deduplicate within the document: identical text yields an identical id,
    # and writing the same id twice in one batch is wasted work.
    by_id: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in chunks}

    stored_ids = await store.existing_chunk_ids(document.document_id)
    new_ids = by_id.keys() - stored_ids
    stale_ids = stored_ids - by_id.keys()

    embedded = await _embed_and_write(
        [by_id[chunk_id] for chunk_id in sorted(new_ids)],
        embedder=embedder,
        store=store,
    )
    deleted = await store.delete_chunks(sorted(stale_ids)) if stale_ids else 0

    return DocumentReport(
        source_uri=document.source_uri,
        document_id=document.document_id,
        total_chunks=len(by_id),
        embedded=embedded,
        skipped=len(by_id) - len(new_ids),
        deleted=deleted,
    )


async def _embed_and_write(
    chunks: Sequence[Chunk],
    *,
    embedder: EmbeddingProvider,
    store: VectorStore,
) -> int:
    if not chunks:
        return 0

    vectors = await embedder.embed_documents([c.text for c in chunks])
    result = await store.upsert(
        [
            EmbeddedChunk(chunk=chunk, embedding=vector, embedding_model=embedder.model)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    )
    return result.written


async def ingest_path(
    path: Path,
    *,
    embedder: EmbeddingProvider,
    store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestReport:
    """Ingest every supported document under ``path``."""
    reports = [
        await ingest_document(
            document,
            embedder=embedder,
            store=store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for document in load_corpus(path)
    ]
    return IngestReport(documents=reports)
