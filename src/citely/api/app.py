"""FastAPI application factory.

A factory rather than a module-level ``app``: tests build an instance with
their own settings, and nothing is constructed at import time.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from citely import __version__
from citely.api.deps import AppState
from citely.api.routes import router
from citely.api.schemas import ErrorResponse
from citely.config import Settings, get_settings
from citely.errors import (
    CitelyError,
    ConfigurationError,
    EmbeddingMismatchError,
    IngestionError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RetrievalError,
    StoreError,
)
from citely.logging import configure_logging, get_logger
from citely.rag.guardrails import InvalidQueryError

log = get_logger("citely.api")

#: Errors are mapped by type, so a new error class gets a sensible status by
#: inheritance rather than by remembering to update a route.
_STATUS_BY_ERROR: list[tuple[type[Exception], int]] = [
    (InvalidQueryError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (IngestionError, status.HTTP_400_BAD_REQUEST),
    (ConfigurationError, status.HTTP_500_INTERNAL_SERVER_ERROR),
    (EmbeddingMismatchError, status.HTTP_409_CONFLICT),
    (ProviderAuthError, status.HTTP_502_BAD_GATEWAY),
    (ProviderRateLimitError, status.HTTP_429_TOO_MANY_REQUESTS),
    (ProviderTimeoutError, status.HTTP_504_GATEWAY_TIMEOUT),
    (ProviderError, status.HTTP_502_BAD_GATEWAY),
    (RetrievalError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (StoreError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (CitelyError, status.HTTP_500_INTERNAL_SERVER_ERROR),
]


def _status_for(exc: Exception) -> int:
    for error_type, code in _STATUS_BY_ERROR:
        if isinstance(exc, error_type):
            return code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def create_app(settings: Settings | None = None, state: AppState | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Overrides the environment-derived configuration.
        state: Pre-built providers and store. Injecting them lets tests run the
            real app with stub providers, instead of patching module internals.
            Production passes neither argument.
    """
    resolved = settings or get_settings()
    configure_logging(level=resolved.log_level, fmt=resolved.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Providers are built here, not per request: one HTTP client, one TLS
        # handshake, and a startup failure surfaces before traffic arrives.
        app.state.citely = state or AppState.build(resolved)
        log.info(
            "startup",
            llm_provider=resolved.llm_provider,
            embedding_provider=resolved.embedding_provider,
            vector_store=resolved.vector_store,
        )
        try:
            yield
        finally:
            await app.state.citely.aclose()
            log.info("shutdown")

    app = FastAPI(
        title="citely",
        version=__version__,
        summary="Retrieval-Augmented Generation as a Service, with grounded, cited answers.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def bind_request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach a request id to every log line produced while serving it."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(CitelyError)
    async def handle_citely_error(_request: Request, exc: CitelyError) -> JSONResponse:
        code = _status_for(exc)
        log.warning("request.failed", code=exc.code, status=code, error=str(exc))
        return JSONResponse(
            status_code=code,
            content=ErrorResponse(code=exc.code, message=str(exc)).model_dump(),
        )

    @app.exception_handler(InvalidQueryError)
    async def handle_invalid_query(_request: Request, exc: InvalidQueryError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=ErrorResponse(code="invalid_query", message=str(exc)).model_dump(),
        )

    app.include_router(router)
    return app
