"""Retrieval.

A narrow protocol on purpose: hybrid search, reranking or query expansion can be
added later as another implementation without the answerer changing at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from citely.errors import CitelyError, RetrievalError
from citely.models import RetrievalResult
from citely.tracing import aspan

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
        async with aspan("retrieval", k=k or self._top_k) as trace:
            try:
                embedding = await self._embedder.embed_query(query)
                hits = await self._store.search(embedding, k=k or self._top_k)
            except CitelyError:
                # Provider and store errors already carry an accurate code, and
                # the API maps them to distinct statuses. Re-wrapping a 502-class
                # provider failure as a 503 retrieval failure would lose that.
                raise
            except Exception as exc:
                raise RetrievalError(f"retrieval failed: {exc}") from exc

            # Filtering here rather than in the store keeps the threshold's
            # meaning in one place, independent of how each backend scores.
            kept = [hit for hit in hits if hit.score >= self._min_score]
            trace.update(
                hits=len(hits),
                kept=len(kept),
                dropped_below_threshold=len(hits) - len(kept),
                top_score=round(max((h.score for h in kept), default=0.0), 4),
            )
            return RetrievalResult(query=query, chunks=kept)
