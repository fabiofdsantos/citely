"""Chroma-backed vector store.

Chroma is the zero-setup default: an embedded, on-disk database with no service
to run, so ``citely ingest`` works on a clean checkout. The pgvector store
implements the same protocol for deployments that already run Postgres.

Chroma's client is synchronous. Rather than pretend otherwise, every call is
pushed to a worker thread with :func:`asyncio.to_thread`, which keeps the event
loop free without inventing a fake async API.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from citely.errors import EmbeddingMismatchError, StoreError
from citely.models import Chunk, ScoredChunk
from citely.stores.base import StoreHealth, UpsertResult

if TYPE_CHECKING:
    from citely.config import Settings
    from citely.models import EmbeddedChunk

#: Collection metadata keys. Namespaced to avoid colliding with Chroma's own
#: ``hnsw:*`` settings.
_MODEL_KEY = "citely_embedding_model"
_DIMS_KEY = "citely_dimensions"

#: Cosine distance ranges over [0, 2]; map it onto a [0, 1] similarity so that
#: score thresholds mean the same thing regardless of the backend.
_MAX_COSINE_DISTANCE = 2.0


def _to_similarity(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance / _MAX_COSINE_DISTANCE))


class ChromaVectorStore:
    """Satisfies :class:`~citely.stores.base.VectorStore`."""

    backend = "chroma"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._collection_name = settings.collection_name
        self._client = client or chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            # Cosine, not Chroma's default L2: embeddings from every provider we
            # support are meant to be compared by angle, not magnitude.
            metadata={"hnsw:space": "cosine"},
        )

        # Cached from collection metadata so the embedding-space guard survives
        # a process restart without re-reading on every write.
        stored = self._collection.metadata or {}
        self._embedding_model: str | None = stored.get(_MODEL_KEY)
        self._dimensions: int | None = stored.get(_DIMS_KEY)

    # -- Writes ------------------------------------------------------------
    async def upsert(self, chunks: Sequence[EmbeddedChunk]) -> UpsertResult:
        if not chunks:
            return UpsertResult(written=0)

        self._guard_embedding_space(chunks)
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[c.chunk.chunk_id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.chunk.text for c in chunks],
            metadatas=[_to_metadata(c.chunk) for c in chunks],
        )
        return UpsertResult(written=len(chunks))

    def _guard_embedding_space(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Refuse to mix embedding models or dimensionalities in one collection.

        Two vectors from different models are not comparable, but nothing about
        the failure looks like an error: search still returns results, they are
        just meaningless. Recording the model on first write and hard-failing on
        divergence turns a silent quality collapse into a startup error.
        """
        model = chunks[0].embedding_model
        dimensions = chunks[0].dimensions

        if self._embedding_model is None:
            stored = self._collection.metadata or {}
            self._collection.modify(
                metadata={
                    # Chroma rejects any modify() payload containing hnsw:*
                    # settings, since the distance function is fixed at
                    # creation. Carry over only our own keys.
                    **{k: v for k, v in stored.items() if not k.startswith("hnsw:")},
                    _MODEL_KEY: model,
                    _DIMS_KEY: dimensions,
                }
            )
            self._embedding_model, self._dimensions = model, dimensions
            return

        if self._embedding_model != model or self._dimensions != dimensions:
            raise EmbeddingMismatchError(
                f"collection {self._collection_name!r} was built with "
                f"{self._embedding_model} ({self._dimensions}d) but these chunks are "
                f"{model} ({dimensions}d); re-ingest into a new collection or switch back"
            )

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        if not chunk_ids:
            return 0
        await asyncio.to_thread(self._collection.delete, ids=list(chunk_ids))
        return len(chunk_ids)

    async def delete_document(self, document_id: str) -> int:
        ids = await self.existing_chunk_ids(document_id)
        return await self.delete_chunks(sorted(ids))

    # -- Reads -------------------------------------------------------------
    async def existing_chunk_ids(self, document_id: str) -> set[str]:
        result = await asyncio.to_thread(
            self._collection.get,
            where={"document_id": document_id},
            include=[],  # ids only: no vectors or documents over the wire
        )
        return set(result.get("ids") or [])

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[ScoredChunk]:
        try:
            result = await asyncio.to_thread(
                self._collection.query,
                query_embeddings=[list(embedding)],
                n_results=k,
                where=dict(filters) if filters else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise StoreError(f"chroma query failed: {exc}") from exc

        return [
            ScoredChunk(chunk=_to_chunk(document, metadata), score=_to_similarity(distance))
            for document, metadata, distance in zip(
                _first(result, "documents"),
                _first(result, "metadatas"),
                _first(result, "distances"),
                strict=True,
            )
        ]

    async def health(self) -> StoreHealth:
        count = await asyncio.to_thread(self._collection.count)
        return StoreHealth(
            backend=self.backend,
            collection=self._collection_name,
            chunk_count=count,
            embedding_model=self._embedding_model,
            dimensions=self._dimensions,
        )


def _first(result: Mapping[str, Any], key: str) -> list[Any]:
    """Unwrap Chroma's batched query response, which nests one list per query."""
    values = result.get(key) or []
    return list(values[0]) if values else []


def _to_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        **chunk.metadata,
        "document_id": chunk.document_id,
        "index": chunk.index,
        "source_uri": chunk.source_uri,
    }
    if chunk.title is not None:
        metadata["title"] = chunk.title
    return metadata


def _to_chunk(text: str, metadata: Mapping[str, Any]) -> Chunk:
    reserved = {"document_id", "index", "source_uri", "title"}
    title = metadata.get("title")
    return Chunk(
        document_id=str(metadata["document_id"]),
        text=text,
        index=int(metadata["index"]),
        source_uri=str(metadata["source_uri"]),
        title=str(title) if title is not None else None,
        metadata={k: v for k, v in metadata.items() if k not in reserved},
    )
