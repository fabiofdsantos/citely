"""Config-driven construction of vector stores."""

from __future__ import annotations

from typing import TYPE_CHECKING

from citely.errors import ConfigurationError
from citely.stores.base import VectorStore

if TYPE_CHECKING:
    from citely.config import Settings


def build_vector_store(settings: Settings) -> VectorStore:
    """Construct the store named by ``CITELY_VECTOR_STORE``."""
    match settings.vector_store:
        case "chroma":
            from citely.stores.chroma import ChromaVectorStore

            return ChromaVectorStore(settings)
        case "pgvector":
            from citely.stores.pgvector import PgVectorStore

            return PgVectorStore(settings)
        case unknown:  # pragma: no cover - Literal typing makes this unreachable
            raise ConfigurationError(f"unknown vector store: {unknown!r}")
