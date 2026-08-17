"""Retrieval.

A narrow protocol on purpose: hybrid search, reranking or query expansion can be
added later as another implementation without the answerer changing at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from citely.errors import RetrievalError
from citely.models import RetrievalResult

if TYPE_CHECKING:
    from citely.providers.base import EmbeddingProvider
    from citely.stores.base import VectorStore


@runtime_checkable
class Retriever(Protocol):
    """Finds the chunks most likely to support an answer."""

    async def retrieve(self, query: str, *, k: int | None = None) -> RetrievalResult: ...


class VectorRetriever:
    """Dense retrieval: embed the query, search the store, drop weak matches."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        *,
        top_k: int = 6,
        min_score: float = 0.0,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    async def retrieve(self, query: str, *, k: int | None = None) -> RetrievalResult:
        try:
            embedding = await self._embedder.embed_query(query)
            hits = await self._store.search(embedding, k=k or self._top_k)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"retrieval failed: {exc}") from exc

        # Filtering here rather than in the store keeps the threshold's meaning
        # in one place, independent of how each backend scores.
        kept = [hit for hit in hits if hit.score >= self._min_score]
        return RetrievalResult(query=query, chunks=kept)
