"""Chroma store tests, running against a real on-disk Chroma instance.

Fakes prove the pipeline's logic; these prove the backend actually behaves the
way the pipeline assumes — metadata survives a round trip, deletes are scoped,
and mixing embedding models is refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from citely.config import Settings
from citely.errors import EmbeddingMismatchError
from citely.models import Chunk, EmbeddedChunk
from citely.stores.base import VectorStore
from citely.stores.chroma import ChromaVectorStore

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    settings = Settings(
        llm_provider="ollama",
        embedding_provider="ollama",
        chroma_path=tmp_path / "chroma",
        collection_name="test",
    )
    return ChromaVectorStore(settings)


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


def test_satisfies_the_protocol(store: ChromaVectorStore) -> None:
    assert isinstance(store, VectorStore)


async def test_round_trip_preserves_provenance(store: ChromaVectorStore) -> None:
    """Citations are only possible if metadata survives storage intact."""
    await store.upsert([embedded("Article 5 prohibits certain practices.")])

    results = await store.search([1.0, 0.0, 0.0], k=1)

    assert len(results) == 1
    chunk = results[0].chunk
    assert chunk.text == "Article 5 prohibits certain practices."
    assert chunk.source_uri == "data/corpus/doc1.txt"
    assert chunk.title == "EU AI Act"
    assert chunk.document_id == "doc1"
    assert chunk.metadata == {"suffix": ".txt"}


async def test_identical_vectors_score_near_one(store: ChromaVectorStore) -> None:
    await store.upsert([embedded("text", vector=[1.0, 0.0, 0.0])])

    [result] = await store.search([1.0, 0.0, 0.0], k=1)

    assert result.score == pytest.approx(1.0, abs=1e-4)


async def test_orthogonal_vectors_score_lower(store: ChromaVectorStore) -> None:
    await store.upsert([embedded("text", vector=[1.0, 0.0, 0.0])])

    [result] = await store.search([0.0, 1.0, 0.0], k=1)

    assert 0.0 <= result.score < 0.75


async def test_upsert_is_idempotent_by_chunk_id(store: ChromaVectorStore) -> None:
    await store.upsert([embedded("same text")])
    await store.upsert([embedded("same text")])

    health = await store.health()
    assert health.chunk_count == 1


async def test_existing_ids_are_scoped_to_one_document(store: ChromaVectorStore) -> None:
    await store.upsert(
        [
            embedded("first", document_id="doc1"),
            embedded("second", document_id="doc2", vector=[0.0, 1.0, 0.0]),
        ]
    )

    assert len(await store.existing_chunk_ids("doc1")) == 1
    assert len(await store.existing_chunk_ids("doc2")) == 1
    assert await store.existing_chunk_ids("missing") == set()


async def test_delete_document_removes_only_its_chunks(store: ChromaVectorStore) -> None:
    await store.upsert(
        [
            embedded("first", document_id="doc1"),
            embedded("second", document_id="doc2", vector=[0.0, 1.0, 0.0]),
        ]
    )

    deleted = await store.delete_document("doc1")

    assert deleted == 1
    assert (await store.health()).chunk_count == 1


async def test_health_records_the_embedding_space(store: ChromaVectorStore) -> None:
    before = await store.health()
    assert before.chunk_count == 0
    assert before.embedding_model is None

    await store.upsert([embedded("text")])

    after = await store.health()
    assert after.embedding_model == "fake-embed"
    assert after.dimensions == 3


async def test_mixing_embedding_models_is_refused(store: ChromaVectorStore) -> None:
    """Silently mixing spaces degrades retrieval without ever raising."""
    await store.upsert([embedded("text", model="fake-embed")])

    with pytest.raises(EmbeddingMismatchError, match="was built with fake-embed"):
        await store.upsert([embedded("other", model="different-embed")])


async def test_mixing_dimensions_is_refused(store: ChromaVectorStore) -> None:
    await store.upsert([embedded("text", vector=[1.0, 0.0, 0.0])])

    with pytest.raises(EmbeddingMismatchError, match="3d"):
        await store.upsert([embedded("other", vector=[1.0, 0.0, 0.0, 0.0])])


async def test_embedding_space_guard_survives_a_restart(tmp_path: Path) -> None:
    """The guard must hold across processes, not just within one object."""
    settings = Settings(
        llm_provider="ollama",
        embedding_provider="ollama",
        chroma_path=tmp_path / "chroma",
        collection_name="test",
    )
    await ChromaVectorStore(settings).upsert([embedded("text", model="fake-embed")])

    reopened = ChromaVectorStore(settings)

    assert (await reopened.health()).embedding_model == "fake-embed"
    with pytest.raises(EmbeddingMismatchError):
        await reopened.upsert([embedded("other", model="different-embed")])


async def test_empty_upsert_is_a_no_op(store: ChromaVectorStore) -> None:
    assert (await store.upsert([])).written == 0
    assert await store.delete_chunks([]) == 0
