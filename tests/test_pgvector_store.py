"""pgvector store tests against a real Postgres.

Skipped unless ``CITELY_TEST_PGVECTOR_DSN`` points at a database with the
pgvector extension available:

    docker run -d --name citely-pg -p 5433:5432 \\
        -e POSTGRES_USER=citely -e POSTGRES_PASSWORD=citely \\
        -e POSTGRES_DB=citely pgvector/pgvector:pg16

    CITELY_TEST_PGVECTOR_DSN=postgresql://citely:citely@localhost:5433/citely \\
        uv run pytest -m pgvector

They mirror the Chroma suite deliberately: two backends behind one protocol are
only interchangeable if the same assertions hold for both.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from pydantic import SecretStr

from citely.config import Settings
from citely.errors import EmbeddingMismatchError, StoreError
from citely.models import Chunk, EmbeddedChunk
from citely.stores.base import VectorStore
from citely.stores.pgvector import PgVectorStore

_RAW_DSN = os.environ.get("CITELY_TEST_PGVECTOR_DSN")
# Settings declares this as SecretStr. Pydantic would coerce a plain str at
# runtime, but passing the declared type keeps the test honest under mypy.
DSN = SecretStr(_RAW_DSN) if _RAW_DSN else None

pytestmark = [
    pytest.mark.pgvector,
    pytest.mark.skipif(not DSN, reason="CITELY_TEST_PGVECTOR_DSN is not set"),
]


@pytest.fixture
async def store() -> AsyncIterator[PgVectorStore]:
    settings = Settings(
        llm_provider="ollama",
        embedding_provider="ollama",
        vector_store="pgvector",
        pgvector_dsn=DSN,
        collection_name="test_pgvector",
    )
    store = PgVectorStore(settings)
    try:
        yield store
    finally:
        # Drop rather than truncate: dimensionality is part of the schema, so
        # tests that change it need a clean table.
        pool = await store._connection_pool()
        async with pool.connection() as conn:
            await conn.execute("DROP TABLE IF EXISTS citely_test_pgvector")
        await store.aclose()


def embedded(
    text: str,
    *,
    document_id: str = "doc1",
    index: int = 0,
    vector: list[float] | None = None,
    model: str = "fake-embed",
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=Chunk(
            document_id=document_id,
            text=text,
            index=index,
            source_uri=f"data/corpus/{document_id}.txt",
            title="EU AI Act",
            metadata={"suffix": ".txt"},
        ),
        embedding=vector or [1.0, 0.0, 0.0],
        embedding_model=model,
    )


def test_satisfies_the_protocol() -> None:
    settings = Settings(
        llm_provider="ollama",
        embedding_provider="ollama",
        vector_store="pgvector",
        pgvector_dsn=DSN,
    )
    assert isinstance(PgVectorStore(settings), VectorStore)


async def test_round_trip_preserves_provenance(store: PgVectorStore) -> None:
    await store.upsert([embedded("Article 5 prohibits certain practices.")])

    results = await store.search([1.0, 0.0, 0.0], k=1)

    assert len(results) == 1
    chunk = results[0].chunk
    assert chunk.text == "Article 5 prohibits certain practices."
    assert chunk.source_uri == "data/corpus/doc1.txt"
    assert chunk.title == "EU AI Act"
    assert chunk.metadata == {"suffix": ".txt"}


async def test_identical_vectors_score_near_one(store: PgVectorStore) -> None:
    await store.upsert([embedded("text", vector=[1.0, 0.0, 0.0])])

    [result] = await store.search([1.0, 0.0, 0.0], k=1)

    assert result.score == pytest.approx(1.0, abs=1e-4)


async def test_orthogonal_vectors_score_lower(store: PgVectorStore) -> None:
    await store.upsert([embedded("text", vector=[1.0, 0.0, 0.0])])

    [result] = await store.search([0.0, 1.0, 0.0], k=1)

    assert 0.0 <= result.score < 0.75


async def test_upsert_is_idempotent_by_chunk_id(store: PgVectorStore) -> None:
    await store.upsert([embedded("same text")])
    await store.upsert([embedded("same text")])

    assert (await store.health()).chunk_count == 1


async def test_existing_ids_are_scoped_to_one_document(store: PgVectorStore) -> None:
    await store.upsert(
        [
            embedded("first", document_id="doc1"),
            embedded("second", document_id="doc2", vector=[0.0, 1.0, 0.0]),
        ]
    )

    assert len(await store.existing_chunk_ids("doc1")) == 1
    assert await store.existing_chunk_ids("missing") == set()


async def test_delete_document_removes_only_its_chunks(store: PgVectorStore) -> None:
    await store.upsert(
        [
            embedded("first", document_id="doc1"),
            embedded("second", document_id="doc2", vector=[0.0, 1.0, 0.0]),
        ]
    )

    assert await store.delete_document("doc1") == 1
    assert (await store.health()).chunk_count == 1


async def test_health_records_the_embedding_space(store: PgVectorStore) -> None:
    await store.upsert([embedded("text")])

    health = await store.health()

    assert health.backend == "pgvector"
    assert health.embedding_model == "fake-embed"
    assert health.dimensions == 3


async def test_mixing_embedding_models_is_refused(store: PgVectorStore) -> None:
    await store.upsert([embedded("text", model="fake-embed")])

    with pytest.raises(EmbeddingMismatchError, match="was built with fake-embed"):
        await store.upsert([embedded("other", model="different-embed")])


async def test_empty_operations_are_no_ops(store: PgVectorStore) -> None:
    assert (await store.upsert([])).written == 0
    assert await store.delete_chunks([]) == 0


async def test_queries_before_first_ingest_return_empty(store: PgVectorStore) -> None:
    """A fresh database has no table yet; that is a normal first-run state."""
    assert await store.search([1.0, 0.0, 0.0], k=5) == []
    assert await store.existing_chunk_ids("doc1") == set()
    assert (await store.health()).chunk_count == 0


def test_unsafe_collection_names_are_rejected() -> None:
    """The collection name becomes a table name and cannot be parameterised."""
    with pytest.raises(StoreError, match="not a valid table name"):
        PgVectorStore(
            Settings(
                llm_provider="ollama",
                embedding_provider="ollama",
                vector_store="pgvector",
                pgvector_dsn=DSN,
                collection_name="chunks; DROP TABLE users--",
            )
        )
