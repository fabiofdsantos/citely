"""Provider tests.

Every test here runs offline against injected fake clients. The value being
protected is the translation layer: SDK shapes in, citely types out, with no
provider exception ever escaping. Tests that hit the real APIs live in
``test_providers_live.py`` behind the ``live`` marker.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx
import openai
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from citely.config import Settings
from citely.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from citely.providers import EmbeddingProvider, LLMProvider, Message
from citely.providers.anthropic import AnthropicChatProvider
from citely.providers.openai import OpenAIChatProvider, OpenAIEmbeddingProvider
from citely.providers.registry import build_embedding_provider, build_llm_provider

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


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
# Protocol conformance
# --------------------------------------------------------------------------


def test_backends_satisfy_the_protocols(settings: Settings) -> None:
    """Structural typing means conformance is never declared, so assert it."""
    chat = OpenAIChatProvider(settings, client=as_openai(FakeOpenAIClient()))
    embedder = OpenAIEmbeddingProvider(settings, client=as_openai(FakeOpenAIClient()))
    claude = AnthropicChatProvider(settings, client=as_anthropic(FakeAnthropicClient()))

    assert isinstance(chat, LLMProvider)
    assert isinstance(claude, LLMProvider)
    assert isinstance(embedder, EmbeddingProvider)


# --------------------------------------------------------------------------
# OpenAI chat
# --------------------------------------------------------------------------


class TestOpenAIChat:
    async def test_normalises_the_response(self, settings: Settings) -> None:
        client = FakeOpenAIClient(chat_response=openai_chat_response("grounded answer"))
        provider = OpenAIChatProvider(settings, client=as_openai(client))

        result = await provider.complete(
            system="be grounded", messages=[Message(role="user", content="hi")]
        )

        assert result.text == "grounded answer"
        assert result.usage.total_tokens == 18
        assert not result.truncated

    async def test_system_prompt_becomes_the_first_message(self, settings: Settings) -> None:
        client = FakeOpenAIClient(chat_response=openai_chat_response())
        provider = OpenAIChatProvider(settings, client=as_openai(client))

        await provider.complete(system="RULES", messages=[Message(role="user", content="hi")])

        sent = client.calls[0]["messages"]
        assert sent[0] == {"role": "system", "content": "RULES"}
        assert sent[1] == {"role": "user", "content": "hi"}

    async def test_truncation_is_detectable(self, settings: Settings) -> None:
        """Structured-output parsing needs to know the model ran out of room."""
        client = FakeOpenAIClient(chat_response=openai_chat_response(finish_reason="length"))
        provider = OpenAIChatProvider(settings, client=as_openai(client))

        assert (
            await provider.complete(system="s", messages=[Message(role="user", content="hi")])
        ).truncated

    async def test_missing_choices_is_a_response_error(self, settings: Settings) -> None:
        empty = SimpleNamespace(model="gpt-4o-mini", choices=[], usage=None)
        provider = OpenAIChatProvider(
            settings, client=as_openai(FakeOpenAIClient(chat_response=empty))
        )

        with pytest.raises(ProviderResponseError):
            await provider.complete(system="s", messages=[Message(role="user", content="hi")])


# --------------------------------------------------------------------------
# OpenAI embeddings
# --------------------------------------------------------------------------


class TestOpenAIEmbeddings:
    async def test_empty_input_makes_no_request(self, settings: Settings) -> None:
        client = FakeOpenAIClient()
        provider = OpenAIEmbeddingProvider(settings, client=as_openai(client))

        assert await provider.embed_documents([]) == []
        assert client.calls == []

    async def test_batches_large_inputs(self, settings: Settings) -> None:
        """600 chunks must not become one oversized request."""
        client = FakeOpenAIClient(
            embedding_response=lambda kwargs: openai_embedding_response(len(kwargs["input"]))
        )
        provider = OpenAIEmbeddingProvider(settings, client=as_openai(client))

        vectors = await provider.embed_documents([f"chunk {i}" for i in range(600)])

        assert len(vectors) == 600
        assert [len(call["input"]) for call in client.calls] == [256, 256, 88]

    async def test_results_are_realigned_by_index(self, settings: Settings) -> None:
        """Out-of-order results would silently pair chunks with wrong vectors."""
        client = FakeOpenAIClient(embedding_response=openai_embedding_response(3, shuffled=True))
        provider = OpenAIEmbeddingProvider(settings, client=as_openai(client))

        vectors = await provider.embed_documents(["a", "b", "c"])

        assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]

    async def test_count_mismatch_is_fatal(self, settings: Settings) -> None:
        client = FakeOpenAIClient(embedding_response=openai_embedding_response(2))
        provider = OpenAIEmbeddingProvider(settings, client=as_openai(client))

        with pytest.raises(ProviderResponseError, match="2 embeddings for 3 inputs"):
            await provider.embed_documents(["a", "b", "c"])

    async def test_embed_query_returns_a_single_vector(self, settings: Settings) -> None:
        client = FakeOpenAIClient(embedding_response=openai_embedding_response(1))
        provider = OpenAIEmbeddingProvider(settings, client=as_openai(client))

        assert await provider.embed_query("what is a high-risk system?") == [0.0, 0.5]


# --------------------------------------------------------------------------
# Anthropic chat
# --------------------------------------------------------------------------


class TestAnthropicChat:
    async def test_system_prompt_is_a_top_level_parameter(self, settings: Settings) -> None:
        """Anthropic has no system role; sending one as a message would fail."""
        client = FakeAnthropicClient(response=anthropic_response())
        provider = AnthropicChatProvider(settings, client=as_anthropic(client))

        await provider.complete(system="RULES", messages=[Message(role="user", content="hi")])

        call = client.calls[0]
        assert call["system"] == "RULES"
        assert call["messages"] == [{"role": "user", "content": "hi"}]

    async def test_joins_text_blocks_and_ignores_others(self, settings: Settings) -> None:
        client = FakeAnthropicClient(
            response=anthropic_response(
                blocks=[
                    SimpleNamespace(type="text", text="part one "),
                    SimpleNamespace(type="tool_use", name="search"),
                    SimpleNamespace(type="text", text="part two"),
                ]
            )
        )
        provider = AnthropicChatProvider(settings, client=as_anthropic(client))

        result = await provider.complete(system="s", messages=[Message(role="user", content="hi")])

        assert result.text == "part one part two"

    async def test_no_text_content_is_a_response_error(self, settings: Settings) -> None:
        client = FakeAnthropicClient(response=anthropic_response(blocks=[]))
        provider = AnthropicChatProvider(settings, client=as_anthropic(client))

        with pytest.raises(ProviderResponseError):
            await provider.complete(system="s", messages=[Message(role="user", content="hi")])

    async def test_empty_message_list_is_rejected_before_the_call(self, settings: Settings) -> None:
        client = FakeAnthropicClient(response=anthropic_response())
        provider = AnthropicChatProvider(settings, client=as_anthropic(client))

        with pytest.raises(ProviderError):
            await provider.complete(system="s", messages=[])
        assert client.calls == []


# --------------------------------------------------------------------------
# Error translation
# --------------------------------------------------------------------------


class TestErrorTranslation:
    """No provider SDK exception may reach a caller."""

    @pytest.mark.parametrize(
        ("raised", "expected"),
        [
            (
                openai.AuthenticationError("bad key", response=http_response(401), body=None),
                ProviderAuthError,
            ),
            (
                openai.RateLimitError("slow down", response=http_response(429), body=None),
                ProviderRateLimitError,
            ),
            (
                openai.APITimeoutError(request=http_request()),
                ProviderTimeoutError,
            ),
            (ValueError("something odd"), ProviderError),
        ],
    )
    async def test_openai_errors_are_translated(
        self, settings: Settings, raised: Exception, expected: type[ProviderError]
    ) -> None:
        provider = OpenAIChatProvider(
            settings, client=as_openai(FakeOpenAIClient(chat_response=raised))
        )

        with pytest.raises(expected):
            await provider.complete(system="s", messages=[Message(role="user", content="hi")])

    @pytest.mark.parametrize(
        ("raised", "expected"),
        [
            (
                anthropic.AuthenticationError("bad key", response=http_response(401), body=None),
                ProviderAuthError,
            ),
            (
                anthropic.RateLimitError("slow down", response=http_response(429), body=None),
                ProviderRateLimitError,
            ),
            (
                anthropic.APITimeoutError(request=http_request()),
                ProviderTimeoutError,
            ),
            (ValueError("something odd"), ProviderError),
        ],
    )
    async def test_anthropic_errors_are_translated(
        self, settings: Settings, raised: Exception, expected: type[ProviderError]
    ) -> None:
        provider = AnthropicChatProvider(
            settings, client=as_anthropic(FakeAnthropicClient(response=raised))
        )

        with pytest.raises(expected):
            await provider.complete(system="s", messages=[Message(role="user", content="hi")])


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


class TestRegistry:
    def test_builds_the_configured_chat_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("OPENAI_API_KEY", "k")

        assert isinstance(
            build_llm_provider(Settings(llm_provider="anthropic")), AnthropicChatProvider
        )
        assert isinstance(build_llm_provider(Settings(llm_provider="openai")), OpenAIChatProvider)

    def test_builds_the_configured_embedding_provider(self, settings: Settings) -> None:
        assert isinstance(build_embedding_provider(settings), OpenAIEmbeddingProvider)
