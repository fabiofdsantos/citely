"""Postgres + pgvector store.

The production path. Chroma is the zero-setup default, but a service that
already runs Postgres should not also operate a second database: pgvector gives
transactional writes, real backups, and one thing to monitor.

Implementation notes:

* Raw SQL over psycopg3's async pool. An ORM would add a dependency, a
  migration tool, and a mapping layer for four queries.
* Vectors are sent as pgvector's text literal (``'[1,2,3]'``) and cast, which
  avoids depending on the ``pgvector`` Python package for binary adaptation.
* The schema is created on first use. For a single-table store this is simpler
  and less error-prone than shipping a migration runner; if the schema grows,
  that trade should be revisited.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from citely.errors import EmbeddingMismatchError, StoreError
from citely.models import Chunk, ScoredChunk
from citely.stores.base import StoreHealth, UpsertResult

if TYPE_CHECKING:
    from citely.config import Settings
    from citely.models import EmbeddedChunk

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table} (
    chunk_id        TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL,
    source_uri      TEXT NOT NULL,
    title           TEXT,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding       VECTOR({dimensions}) NOT NULL,
    embedding_model TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Incremental ingestion looks up every chunk of one document on each run, so
-- this index is on the hot path rather than a nice-to-have.
CREATE INDEX IF NOT EXISTS {table}_document_id_idx ON {table} (document_id);

-- HNSW, not IVFFlat, with cosine distance to match the Chroma backend's metric
-- so a score threshold means the same thing on either store.
--
-- The index is created before any rows exist, and an IVFFlat index built on an
-- empty table has untrained lists: queries then return few or no rows until it
-- is rebuilt, which looks like broken retrieval rather than a missing REINDEX.
-- HNSW builds incrementally and needs no training set, so it is correct from
-- the first insert. It costs more to build and more memory; for a corpus that
-- fits in one Postgres, that is the right trade.
CREATE INDEX IF NOT EXISTS {table}_embedding_idx
    ON {table} USING hnsw (embedding vector_cosine_ops);
"""


def _to_vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in embedding) + "]"


def _to_similarity(distance: float) -> float:
    """pgvector's ``<=>`` returns cosine distance in [0, 2]."""
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


class PgVectorStore:
    """Satisfies :class:`~citely.stores.base.VectorStore`."""

    backend = "pgvector"

    def __init__(self, settings: Settings) -> None:
        if settings.pgvector_dsn is None:  # pragma: no cover - config prevents this
            raise StoreError("CITELY_PGVECTOR_DSN is not configured")

        self._dsn = settings.pgvector_dsn.get_secret_value()
        # The collection name becomes the table name, so it is validated rather
        # than interpolated blindly: table names cannot be bound as parameters.
        self._table = _safe_identifier(settings.collection_name)
        self._pool: AsyncConnectionPool | None = None
        self._ready = False

    async def _connection_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            self._pool = AsyncConnectionPool(self._dsn, min_size=1, max_size=8, open=False)
            await self._pool.open(wait=True, timeout=10)
        return self._pool

    async def _ensure_schema(self, conn: AsyncConnection[Any], dimensions: int) -> None:
        if self._ready:
            return
        await conn.execute(_SCHEMA.format(table=self._table, dimensions=dimensions))
        self._ready = True

    # -- Writes ------------------------------------------------------------
    async def upsert(self, chunks: Sequence[EmbeddedChunk]) -> UpsertResult:
        if not chunks:
            return UpsertResult(written=0)

        model, dimensions = chunks[0].embedding_model, chunks[0].dimensions
        pool = await self._connection_pool()

        try:
            async with pool.connection() as conn:
                await self._ensure_schema(conn, dimensions)
                await self._guard_embedding_space(conn, model, dimensions)

                await conn.cursor().executemany(
                    f"""
                    INSERT INTO {self._table} (
                        chunk_id, document_id, source_uri, title, chunk_index,
                        content, metadata, embedding, embedding_model
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model
                    """,  # noqa: S608 - table name is validated, never user input
                    [
                        (
                            c.chunk.chunk_id,
                            c.chunk.document_id,
                            c.chunk.source_uri,
                            c.chunk.title,
                            c.chunk.index,
                            c.chunk.text,
                            json.dumps(c.chunk.metadata),
                            _to_vector_literal(c.embedding),
                            c.embedding_model,
                        )
                        for c in chunks
                    ],
                )
        except EmbeddingMismatchError:
            raise
        except Exception as exc:
            raise StoreError(f"pgvector upsert failed: {exc}") from exc

        return UpsertResult(written=len(chunks))

    async def _guard_embedding_space(
        self, conn: AsyncConnection[Any], model: str, dimensions: int
    ) -> None:
        """Refuse to mix embedding models, exactly as the Chroma store does."""
        cursor = await conn.execute(
            f"SELECT embedding_model FROM {self._table} LIMIT 1"  # noqa: S608 - validated identifier
        )
        row = await cursor.fetchone()
        if row and row[0] != model:
            raise EmbeddingMismatchError(
                f"table {self._table!r} was built with {row[0]} but these chunks "
                f"are {model} ({dimensions}d); re-ingest into a new collection"
            )

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        if not chunk_ids:
            return 0
        pool = await self._connection_pool()
        try:
            async with pool.connection() as conn:
                cursor = await conn.execute(
                    f"DELETE FROM {self._table} WHERE chunk_id = ANY(%s)",  # noqa: S608
                    (list(chunk_ids),),
                )
                return cursor.rowcount
        except Exception as exc:
            raise StoreError(f"pgvector delete failed: {exc}") from exc

    async def delete_document(self, document_id: str) -> int:
        pool = await self._connection_pool()
        try:
            async with pool.connection() as conn:
                cursor = await conn.execute(
                    f"DELETE FROM {self._table} WHERE document_id = %s",  # noqa: S608
                    (document_id,),
                )
                return cursor.rowcount
        except Exception as exc:
            raise StoreError(f"pgvector delete failed: {exc}") from exc

    # -- Reads -------------------------------------------------------------
    async def existing_chunk_ids(self, document_id: str) -> set[str]:
        pool = await self._connection_pool()
        try:
            async with pool.connection() as conn:
                cursor = await conn.execute(
                    f"SELECT chunk_id FROM {self._table} WHERE document_id = %s",  # noqa: S608
                    (document_id,),
                )
                return {row[0] for row in await cursor.fetchall()}
        except Exception as exc:
            # A missing table means nothing has been ingested yet, which is a
            # legitimate state on a first run rather than a failure.
            if "does not exist" in str(exc):
                return set()
            raise StoreError(f"pgvector lookup failed: {exc}") from exc

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[ScoredChunk]:
        pool = await self._connection_pool()
        where, params = "", [_to_vector_literal(embedding)]
        if filters:
            where = " WHERE metadata @> %s::jsonb"
            params.append(json.dumps(dict(filters)))

        try:
            async with pool.connection() as conn:
                cursor = conn.cursor(row_factory=dict_row)
                await cursor.execute(
                    f"""
                    SELECT chunk_id, document_id, source_uri, title, chunk_index,
                           content, metadata, embedding <=> %s::vector AS distance
                    FROM {self._table}{where}
                    ORDER BY distance
                    LIMIT {int(k)}
                    """,  # noqa: S608 - identifier validated, k coerced to int
                    params,
                )
                rows = await cursor.fetchall()
        except Exception as exc:
            if "does not exist" in str(exc):
                return []
            raise StoreError(f"pgvector query failed: {exc}") from exc

        return [
            ScoredChunk(
                chunk=Chunk(
                    document_id=row["document_id"],
                    text=row["content"],
                    index=row["chunk_index"],
                    source_uri=row["source_uri"],
                    title=row["title"],
                    metadata=row["metadata"] or {},
                ),
                score=_to_similarity(float(row["distance"])),
            )
            for row in rows
        ]

    async def health(self) -> StoreHealth:
        pool = await self._connection_pool()
        try:
            async with pool.connection() as conn:
                cursor = await conn.execute(
                    f"""
                    SELECT COUNT(*) AS n,
                           MIN(embedding_model) AS model,
                           MIN(vector_dims(embedding)) AS dims
                    FROM {self._table}
                    """  # noqa: S608 - validated identifier
                )
                row = await cursor.fetchone()
        except Exception as exc:
            if "does not exist" in str(exc):
                row = (0, None, None)
            else:
                raise StoreError(f"pgvector health check failed: {exc}") from exc

        count, model, dims = row if row else (0, None, None)
        return StoreHealth(
            backend=self.backend,
            collection=self._table,
            chunk_count=count or 0,
            embedding_model=model,
            dimensions=dims,
        )

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


def _safe_identifier(name: str) -> str:
    """Validate a name used as a SQL identifier.

    Table names cannot be passed as bound parameters, so the only safe options
    are a strict allowlist or psycopg's ``sql.Identifier``. An allowlist is used
    here because the value comes from configuration, and a startup error is a
    better outcome than a quoted-but-bizarre table name.
    """
    if not name.replace("_", "").isalnum() or name[0].isdigit():
        raise StoreError(
            f"collection name {name!r} is not a valid table name: use letters, "
            "digits and underscores, not starting with a digit"
        )
    return f"citely_{name}"
