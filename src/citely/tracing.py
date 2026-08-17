"""Lightweight tracing hooks.

Deliberately not OpenTelemetry. This is a span-shaped log line — name, duration,
outcome, and whatever fields the caller attaches — which answers the questions
that actually come up ("is retrieval or generation the slow half?", "how often
does verification reject everything?") at a fraction of the dependency weight.

The span field names match OTel conventions, so swapping the implementation for
a real exporter later is a change in this file alone.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator, MutableMapping
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from citely.logging import get_logger

_log = get_logger("citely.trace")


def _emit(name: str, started: float, fields: MutableMapping[str, Any], error: str | None) -> None:
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    event = f"{name}.failed" if error else f"{name}.completed"
    _log.info(event, span=name, duration_ms=duration_ms, error=error, **fields)


@contextmanager
def span(name: str, **fields: Any) -> Iterator[MutableMapping[str, Any]]:
    """Time a block and log the result.

    The yielded dict is mutable so a caller can attach values discovered
    *during* the block — the chunk count, the model that answered — which are
    exactly the fields worth having and are never known up front.
    """
    started = time.perf_counter()
    attributes: MutableMapping[str, Any] = dict(fields)
    try:
        yield attributes
    except Exception as exc:
        _emit(name, started, attributes, error=type(exc).__name__)
        raise
    _emit(name, started, attributes, error=None)


@asynccontextmanager
async def aspan(name: str, **fields: Any) -> AsyncIterator[MutableMapping[str, Any]]:
    """Async counterpart to :func:`span`."""
    started = time.perf_counter()
    attributes: MutableMapping[str, Any] = dict(fields)
    try:
        yield attributes
    except Exception as exc:
        _emit(name, started, attributes, error=type(exc).__name__)
        raise
    _emit(name, started, attributes, error=None)
