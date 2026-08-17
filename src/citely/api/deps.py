"""Application state and dependency wiring.

Providers and the store are built once during startup and reused, rather than
per request: constructing an HTTP client per request leaks connections and
re-does TLS handshakes on every call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from citely.config import Settings
from citely.providers.base import EmbeddingProvider, LLMProvider
from citely.providers.registry import build_embedding_provider, build_llm_provider
from citely.rag.answerer import Answerer
from citely.rag.retriever import VectorRetriever
from citely.stores.base import VectorStore
from citely.stores.registry import build_vector_store


@dataclass(slots=True)
class AppState:
    """Long-lived objects shared by every request."""

    settings: Settings
    embedder: EmbeddingProvider
    llm: LLMProvider
    store: VectorStore
    answerer: Answerer

    @classmethod
    def build(cls, settings: Settings) -> AppState:
        embedder = build_embedding_provider(settings)
        llm = build_llm_provider(settings)
        store = build_vector_store(settings)
        retriever = VectorRetriever(
            embedder, store, top_k=settings.top_k, min_score=settings.min_score
        )
        return cls(
            settings=settings,
            embedder=embedder,
            llm=llm,
            store=store,
            answerer=Answerer(
                retriever,
                llm,
                max_context_tokens=settings.max_context_tokens,
                scope_check=settings.scope_check,
                scope_ignore_terms=settings.scope_ignored_terms,
            ),
        )

    async def aclose(self) -> None:
        await self.embedder.aclose()
        await self.llm.aclose()


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.citely
    return state


StateDep = Annotated[AppState, Depends(get_state)]
