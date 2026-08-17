"""Vector store interface.

The methods here are exactly what incremental ingestion and retrieval need, and
nothing more. In particular ``existing_chunk_ids`` is what makes re-ingestion
cheap: chunk ids are content hashes, so the set of ids already stored for a
document tells us what to skip embedding and what to delete, without reading a
single vector back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from citely.models import EmbeddedChunk, ScoredChunk


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class UpsertResult(_Frozen):
    """What one upsert changed."""

    written: int = Field(ge=0)
    deleted: int = Field(ge=0, default=0)


class StoreHealth(_Frozen):
    """A store's readiness, used by ``/healthz`` and by the CLI."""

    backend: str
    collection: str
    chunk_count: int = Field(ge=0)
    embedding_model: str | None = None
    dimensions: int | None = None


@runtime_checkable
class VectorStore(Protocol):
    """Persistence and similarity search over embedded chunks."""

    async def upsert(self, chunks: Sequence[EmbeddedChunk]) -> UpsertResult:
        """Insert or replace chunks, keyed by chunk id.

        Raises:
            EmbeddingMismatchError: if the chunks were embedded by a different
                model or dimensionality than the collection already holds.
        """
        ...

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[ScoredChunk]:
        """Return the ``k`` nearest chunks, most similar first."""
        ...

    async def existing_chunk_ids(self, document_id: str) -> set[str]:
        """Return the ids currently stored for one document."""
        ...

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        """Delete chunks by id, returning how many were removed."""
        ...

    async def delete_document(self, document_id: str) -> int:
        """Delete every chunk of a document, returning how many were removed."""
        ...

    async def health(self) -> StoreHealth:
        """Describe the collection's current state."""
        ...
