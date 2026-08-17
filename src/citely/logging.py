"""Structured logging.

JSON in production so logs are queryable, human-readable colour in a terminal.
Configured once at process start; everything else just calls
:func:`get_logger`.

The rule this module exists to enforce: log *events with fields*, never
formatted sentences. ``log.info("retrieval.completed", chunks=6, ms=48)`` can be
aggregated and alerted on; ``log.info(f"got {n} chunks in {ms}ms")`` cannot.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def configure_logging(*, level: str = "INFO", fmt: str = "json") -> None:
    """Configure structlog and the standard library's logging together.

    Idempotent: safe to call from the CLI, the API's lifespan, and tests.
    """
    global _configured  # process-wide setup guard
    if _configured:
        return

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level]),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Route stdlib loggers (uvicorn, httpx, chromadb) through the same handler
    # so one process never emits two log formats.
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)
    _configured = True


def reset_logging() -> None:
    """Undo configuration. For tests only."""
    global _configured  # process-wide setup guard
    _configured = False
    structlog.reset_defaults()


def get_logger(name: str = "citely", **initial: Any) -> Any:
    """Return a logger, optionally pre-bound with context fields.

    Without initial fields this stays a lazy proxy that resolves its
    configuration on first use. That matters: ``bind()`` materialises a logger
    against whatever configuration exists *at that moment*, so a module-level
    ``get_logger(...)`` that bound eagerly would freeze the default renderer and
    ignore ``configure_logging`` called later during startup.
    """
    logger = structlog.get_logger(name)
    return logger.bind(**initial) if initial else logger
