"""Anthropic backend for chat.

There is no embedding counterpart here: Anthropic does not offer an embeddings
API, so an Anthropic-for-generation setup pairs with another provider for
vectors. That asymmetry is why the two protocols are separate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import anthropic
from anthropic import AsyncAnthropic

from citely.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from citely.providers.base import Completion, Message, TokenUsage

if TYPE_CHECKING:
    from citely.config import Settings


def _translate(exc: Exception) -> ProviderError:
    """Map SDK exceptions onto citely's error hierarchy."""
    match exc:
        case anthropic.AuthenticationError() | anthropic.PermissionDeniedError():
            return ProviderAuthError(f"Anthropic rejected the credentials: {exc}")
        case anthropic.RateLimitError():
            return ProviderRateLimitError(f"Anthropic rate limit reached: {exc}")
        case anthropic.APITimeoutError():
            return ProviderTimeoutError(f"Anthropic request timed out: {exc}")
        case _:
            return ProviderError(f"Anthropic request failed: {exc}")


class AnthropicChatProvider:
    """Chat via the Messages API. Satisfies :class:`~citely.providers.base.LLMProvider`."""

    def __init__(self, settings: Settings, client: AsyncAnthropic | None = None) -> None:
        self._model = settings.resolved_llm_model
        self._client = client or self._build_client(settings)

    @staticmethod
    def _build_client(settings: Settings) -> AsyncAnthropic:
        if settings.anthropic_api_key is None:  # pragma: no cover - config prevents this
            raise ProviderAuthError("ANTHROPIC_API_KEY is not configured")
        return AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
        )

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        if not messages:
            raise ProviderError("Anthropic requires at least one message")

        try:
            response = await self._client.messages.create(
                model=self._model,
                # Unlike OpenAI, the system prompt is a top-level parameter,
                # not a message with role="system".
                system=system,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise _translate(exc) from exc

        # Content is a list of typed blocks; a response may interleave text with
        # other block types, so filter rather than assuming content[0] is text.
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text and response.stop_reason != "max_tokens":
            raise ProviderResponseError("Anthropic returned no text content")

        return Completion(
            text=text,
            model=response.model,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            stop_reason=response.stop_reason,
        )

    async def aclose(self) -> None:
        await self._client.close()
