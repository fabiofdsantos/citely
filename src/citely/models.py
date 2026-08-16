"""Domain models.

These are the values that move between ingestion, storage, retrieval and
generation. They are immutable (``frozen=True``) because every one of them is a
value object: a chunk with a different body is a different chunk, not a mutated
one. Wire-format request/response models live in ``citely.api.schemas`` instead,
so the HTTP contract can evolve without dragging the domain with it.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Length of the truncated SHA-256 digests used for ids. 16 hex chars is 64
#: bits: collision-safe far beyond any realistic corpus, and short enough to
#: read in logs and citations.
_ID_LENGTH = 16

Metadata = dict[str, str | int | float | bool]


def _digest(*parts: str) -> str:
    joined = "\x00".join(parts).encode()
    return hashlib.sha256(joined).hexdigest()[:_ID_LENGTH]


def document_id_for(source_uri: str) -> str:
    """Derive a stable document id from its source location."""
    return _digest(source_uri)


def chunk_id_for(document_id: str, text: str) -> str:
    """Derive a content-addressed chunk id.

    Deliberately hashes the *content* and not the chunk's position: editing one
    paragraph then leaves every other chunk's id untouched, so re-ingestion only
    re-embeds what actually changed. The trade-off is that two byte-identical
    chunks in the same document collapse into one row, which is acceptable —
    they are interchangeable as citation targets.
    """
    return _digest(document_id, text)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RawDocument(_Frozen):
    """A source document as loaded, before chunking."""

    source_uri: str = Field(min_length=1, description="Where this came from: path or URL.")
    text: str = Field(min_length=1)
    title: str | None = None
    metadata: Metadata = Field(default_factory=dict)

    @property
    def document_id(self) -> str:
        return document_id_for(self.source_uri)


class Chunk(_Frozen):
    """A retrievable unit of a document."""

    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    index: int = Field(ge=0, description="Position within the document, for ordering and display.")
    source_uri: str = Field(min_length=1)
    title: str | None = None
    metadata: Metadata = Field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return chunk_id_for(self.document_id, self.text)


class EmbeddedChunk(_Frozen):
    """A chunk plus its vector, tagged with the model that produced it.

    The model name travels with the vector so the store can refuse to mix
    embedding spaces, which otherwise fails silently and looks like bad
    retrieval rather than a configuration bug.
    """

    chunk: Chunk
    embedding: list[float] = Field(min_length=1)
    embedding_model: str = Field(min_length=1)

    @property
    def dimensions(self) -> int:
        return len(self.embedding)


class ScoredChunk(_Frozen):
    """A retrieved chunk with its similarity score, normalised to [0, 1]."""

    chunk: Chunk
    score: Annotated[float, Field(ge=0.0, le=1.0)]


class RetrievalResult(_Frozen):
    """Everything retrieval produced for one query."""

    query: str = Field(min_length=1)
    chunks: list[ScoredChunk] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def top_score(self) -> float:
        return max((c.score for c in self.chunks), default=0.0)


class Citation(_Frozen):
    """A claim's link back to the chunk that supports it.

    ``quote`` is verified to appear verbatim in the cited chunk before an answer
    is returned; a citation that cannot be verified invalidates the answer.
    """

    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    title: str | None = None


class Answer(_Frozen):
    """The result of a query: either a grounded, cited answer or a refusal."""

    query: str = Field(min_length=1)
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    model: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Enforce the core product invariant.

        An answer either refuses, or it cites. There is no third state, because
        an uncited assertion is exactly the failure mode this service exists to
        prevent.
        """
        if self.refused:
            if not self.refusal_reason:
                raise ValueError("a refused answer must carry a refusal_reason")
            if self.citations:
                raise ValueError("a refused answer must not carry citations")
        elif not self.citations:
            raise ValueError("a non-refused answer must cite at least one source")
        return self
