"""HTTP request and response models.

Separate from ``citely.models`` on purpose: the domain is free to change shape,
while the wire contract is a promise to clients. Anything a caller can see is
declared here, which also means the OpenAPI schema is complete without
annotating routes by hand.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from citely.models import Answer


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=1000,
        description="The question to answer from the indexed corpus.",
        examples=["What does Article 5 prohibit?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Chunks to retrieve. Defaults to the server's configured value.",
    )


class CitationResponse(BaseModel):
    """A claim's link back to its source, with the quote that supports it."""

    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1, description="Matches the [n] markers in the answer text.")
    chunk_id: str
    source_uri: str
    title: str | None = None
    quote: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    refused: bool = Field(description="True when the corpus could not support a grounded answer.")
    refusal_reason: str | None = None
    model: str | None = None

    @classmethod
    def from_answer(cls, answer: Answer) -> QueryResponse:
        return cls(
            question=answer.query,
            answer=answer.text,
            citations=[
                CitationResponse(
                    number=number,
                    chunk_id=citation.chunk_id,
                    source_uri=citation.source_uri,
                    title=citation.title,
                    quote=citation.quote,
                )
                for number, citation in enumerate(answer.citations, start=1)
            ],
            refused=answer.refused,
            refusal_reason=answer.refusal_reason,
            model=answer.model,
        )


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(
        default=None,
        description="File or directory to ingest. Defaults to the server's corpus path.",
    )


class DocumentIngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_uri: str
    total_chunks: int
    embedded: int
    skipped: int
    deleted: int


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentIngestResponse] = Field(default_factory=list)
    embedded: int
    skipped: int
    deleted: int
    unchanged: bool = Field(description="True when the run was a no-op.")


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    version: str
    backend: str
    collection: str
    chunk_count: int
    embedding_model: str | None = None
    dimensions: int | None = None


class ErrorResponse(BaseModel):
    """The shape of every non-2xx body, so clients can branch on one field."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Stable machine-readable error code.", examples=["store_error"])
    message: str
