"""HTTP routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, status

from citely import __version__
from citely.api.deps import StateDep
from citely.api.schemas import (
    DocumentIngestResponse,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from citely.ingest.pipeline import ingest_path
from citely.logging import get_logger

router = APIRouter()
log = get_logger("citely.api")

_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
    status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse},
}


@router.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz(state: StateDep) -> HealthResponse:
    """Report service and index state.

    Deliberately touches the store: a process that is up but cannot reach its
    index is not healthy, and reporting otherwise defeats the point of a probe.
    """
    health = await state.store.health()
    return HealthResponse(
        status="ok",
        version=__version__,
        backend=health.backend,
        collection=health.collection,
        chunk_count=health.chunk_count,
        embedding_model=health.embedding_model,
        dimensions=health.dimensions,
    )


@router.post("/query", response_model=QueryResponse, responses=_ERRORS, tags=["rag"])
async def query(request: QueryRequest, state: StateDep) -> QueryResponse:
    """Answer a question from the corpus, with verified citations.

    A refusal is a 200, not an error: "the corpus cannot support an answer" is a
    valid, expected result that clients must render, not an exception.
    """
    answer = await state.answerer.answer(request.question, k=request.top_k)
    log.info(
        "query.completed",
        refused=answer.refused,
        citations=len(answer.citations),
        model=answer.model,
    )
    return QueryResponse.from_answer(answer)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses=_ERRORS,
    status_code=status.HTTP_200_OK,
    tags=["corpus"],
)
async def ingest(request: IngestRequest, state: StateDep) -> IngestResponse:
    """Load, chunk, embed and index a corpus. Safe to call repeatedly."""
    corpus = Path(request.path) if request.path else state.settings.corpus_path
    report = await ingest_path(
        corpus,
        embedder=state.embedder,
        store=state.store,
        chunk_size=state.settings.chunk_size,
        chunk_overlap=state.settings.chunk_overlap,
    )
    log.info(
        "ingest.completed",
        documents=len(report.documents),
        embedded=report.embedded,
        skipped=report.skipped,
        deleted=report.deleted,
    )
    return IngestResponse(
        documents=[
            DocumentIngestResponse(
                source_uri=d.source_uri,
                total_chunks=d.total_chunks,
                embedded=d.embedded,
                skipped=d.skipped,
                deleted=d.deleted,
            )
            for d in report.documents
        ],
        embedded=report.embedded,
        skipped=report.skipped,
        deleted=report.deleted,
        unchanged=report.unchanged,
    )
