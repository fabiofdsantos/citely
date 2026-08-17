"""Shared test doubles for the provider layer.

Kept out of the test modules themselves so both the OpenAI and Ollama suites can
use them without importing each other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from citely.errors import EmbeddingMismatchError
from citely.models import EmbeddedChunk, ScoredChunk
from citely.providers.base import Completion, TokenUsage
from citely.stores.base import StoreHealth, UpsertResult


class FakeOpenAIClient:
    """Records calls and replays canned responses or raises."""

    def __init__(self, chat_response: Any = None, embedding_response: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self._chat_response = chat_response
        self._embedding_response = embedding_response
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat))
        self.embeddings = SimpleNamespace(create=self._create_embedding)

    async def _create_chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._chat_response, Exception):
            raise self._chat_response
        return self._chat_response

    async def _create_embedding(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._embedding_response, Exception):
            raise self._embedding_response
        if callable(self._embedding_response):
            return self._embedding_response(kwargs)
        return self._embedding_response

    async def close(self) -> None:
        self.closed = True


class FakeAnthropicClient:
    def __init__(self, response: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self._response = response
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    async def close(self) -> None:
        self.closed = True


def openai_chat_response(text: str = "hello", finish_reason: str = "stop") -> Any:
    return SimpleNamespace(
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


def openai_embedding_response(count: int, *, shuffled: bool = False) -> Any:
    data = [SimpleNamespace(index=i, embedding=[float(i), 0.5]) for i in range(count)]
    if shuffled:
        data.reverse()
    return SimpleNamespace(data=data)


def anthropic_response(blocks: list[Any] | None = None, stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(
        model="claude-sonnet-5",
        content=blocks if blocks is not None else [SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        stop_reason=stop_reason,
    )


def http_response(status: int) -> Any:
    """A minimal HTTP response for constructing SDK exceptions.

    Typed as ``Any`` on purpose: the OpenAI SDK vendors ``httpx2`` while the
    Anthropic SDK uses ``httpx``, so a concrete annotation would satisfy one and
    fail the other. Pinning the tests to either SDK's private HTTP dependency
    would be worse than this cast.
    """
    return httpx.Response(status_code=status, request=http_request())


def http_request() -> Any:
    """See :func:`http_response` for why this is untyped."""
    return httpx.Request("POST", "https://example.test")


def as_openai(client: FakeOpenAIClient) -> AsyncOpenAI:
    return cast(AsyncOpenAI, client)


def as_anthropic(client: FakeAnthropicClient) -> AsyncAnthropic:
    return cast(AsyncAnthropic, client)


# --------------------------------------------------------------------------
# Ingestion doubles
# --------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic embeddings, with a call counter.

    The counter is the point: incremental ingestion is a claim about how many
    times this gets called, so tests assert on it directly.
    """

    def __init__(self, model: str = "fake-embed", dimensions: int = 4) -> None:
        self._model = model
        self._dimensions = dimensions
        self.embedded_texts: list[str] = []
        self.calls = 0

    @property
    def model(self) -> str:
        return self._model

    def _vector(self, text: str) -> list[float]:
        # A stable pseudo-embedding: similar strings share leading components.
        seed = [float(ord(c)) for c in text[: self._dimensions]]
        seed += [0.0] * (self._dimensions - len(seed))
        norm = sum(v * v for v in seed) ** 0.5 or 1.0
        return [v / norm for v in seed]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        self.embedded_texts.extend(texts)
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._vector(text)

    async def aclose(self) -> None:
        return None


class InMemoryVectorStore:
    """A dict-backed store that honours the VectorStore protocol."""

    backend = "memory"

    def __init__(self, collection: str = "test") -> None:
        self.collection = collection
        self.chunks: dict[str, EmbeddedChunk] = {}
        self._model: str | None = None
        self._dimensions: int | None = None

    async def upsert(self, chunks: Sequence[EmbeddedChunk]) -> UpsertResult:
        if not chunks:
            return UpsertResult(written=0)

        model, dimensions = chunks[0].embedding_model, chunks[0].dimensions
        if self._model is None:
            self._model, self._dimensions = model, dimensions
        elif (self._model, self._dimensions) != (model, dimensions):
            raise EmbeddingMismatchError(
                f"collection holds {self._model} ({self._dimensions}d), got {model} ({dimensions}d)"
            )

        for chunk in chunks:
            self.chunks[chunk.chunk.chunk_id] = chunk
        return UpsertResult(written=len(chunks))

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[ScoredChunk]:
        def cosine(a: Sequence[float], b: Sequence[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            return max(0.0, min(1.0, (dot + 1.0) / 2.0))

        candidates = [
            ScoredChunk(chunk=stored.chunk, score=cosine(embedding, stored.embedding))
            for stored in self.chunks.values()
            if not filters or all(stored.chunk.metadata.get(k2) == v for k2, v in filters.items())
        ]
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:k]

    async def existing_chunk_ids(self, document_id: str) -> set[str]:
        return {
            chunk_id
            for chunk_id, stored in self.chunks.items()
            if stored.chunk.document_id == document_id
        }

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if self.chunks.pop(chunk_id, None) is not None:
                removed += 1
        return removed

    async def delete_document(self, document_id: str) -> int:
        return await self.delete_chunks(sorted(await self.existing_chunk_ids(document_id)))

    async def health(self) -> StoreHealth:
        return StoreHealth(
            backend=self.backend,
            collection=self.collection,
            chunk_count=len(self.chunks),
            embedding_model=self._model,
            dimensions=self._dimensions,
        )


class FakeLLM:
    """Replays canned completions and records the prompts it was given."""

    def __init__(self, *responses: str, model: str = "fake-llm") -> None:
        self._responses = list(responses) or ["{}"]
        self._model = model
        self.prompts: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        self.prompts.append((system, messages[-1].content))
        text = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return Completion(
            text=text,
            model=self._model,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def aclose(self) -> None:
        return None
