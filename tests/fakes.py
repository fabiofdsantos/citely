"""Shared test doubles for the provider layer.

Kept out of the test modules themselves so both the OpenAI and Ollama suites can
use them without importing each other.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


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
