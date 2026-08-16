"""Backends for every provider speaking the OpenAI wire format.

That includes OpenAI itself and Ollama, whose ``/v1`` endpoint is
OpenAI-compatible, so one client implementation serves both: only the base URL
and the credential differ. The same classes reach vLLM, LM Studio or a corporate
gateway by pointing ``CITELY_OPENAI_BASE_URL`` at them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import openai
from openai import AsyncOpenAI

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

#: The embeddings endpoint accepts many inputs per request, but oversized
#: batches risk both the request-size limit and losing a whole batch to one
#: transient failure. 256 is a conservative, well-tested middle ground.
_EMBED_BATCH_SIZE = 256


#: The SDK requires a non-empty api_key even when the server ignores it, which
#: local runtimes do. This placeholder keeps the zero-credential path working.
_NO_CREDENTIAL_REQUIRED = "not-needed"


def _translate(exc: Exception, label: str = "OpenAI") -> ProviderError:
    """Map SDK exceptions onto citely's error hierarchy.

    Callers should never need to import ``openai`` to handle a failure; that is
    the whole point of the provider abstraction.
    """
    match exc:
        case openai.AuthenticationError() | openai.PermissionDeniedError():
            return ProviderAuthError(f"{label} rejected the credentials: {exc}")
        case openai.RateLimitError():
            return ProviderRateLimitError(f"{label} rate limit reached: {exc}")
        case openai.APITimeoutError():
            return ProviderTimeoutError(f"{label} request timed out: {exc}")
        case openai.APIConnectionError():
            # By far the most common local failure: the runtime is not running.
            return ProviderError(f"{label} is unreachable at the configured base URL: {exc}")
        case _:
            return ProviderError(f"{label} request failed: {exc}")


def _build_client(settings: Settings, provider: str) -> AsyncOpenAI:
    if provider == "ollama":
        api_key = _NO_CREDENTIAL_REQUIRED
    elif settings.openai_api_key is None:  # pragma: no cover - config validation prevents this
        raise ProviderAuthError("OPENAI_API_KEY is not configured")
    else:
        api_key = settings.openai_api_key.get_secret_value()

    return AsyncOpenAI(
        api_key=api_key,
        base_url=settings.base_url_for(provider),
        timeout=settings.request_timeout_seconds,
        # The SDK already implements bounded exponential backoff with jitter and
        # honours Retry-After. Re-implementing it would be strictly worse.
        max_retries=settings.max_retries,
    )


class OpenAIChatProvider:
    """Chat completions over the OpenAI wire format.

    Satisfies :class:`~citely.providers.base.LLMProvider`.
    """

    def __init__(
        self,
        settings: Settings,
        client: AsyncOpenAI | None = None,
        provider: str = "openai",
    ) -> None:
        self._model = settings.resolved_llm_model
        self._label = provider
        self._client = client or _build_client(settings, provider)

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
        payload: list[dict[str, str]] = [{"role": "system", "content": system}]
        payload += [{"role": m.role, "content": m.content} for m in messages]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=payload,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise _translate(exc, self._label) from exc

        if not response.choices:
            raise ProviderResponseError(f"{self._label} returned no choices")

        choice = response.choices[0]
        usage = response.usage
        return Completion(
            text=choice.message.content or "",
            model=response.model,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            stop_reason=choice.finish_reason,
        )

    async def aclose(self) -> None:
        await self._client.close()


class OpenAIEmbeddingProvider:
    """Embeddings over the OpenAI wire format.

    Satisfies :class:`~citely.providers.base.EmbeddingProvider`.
    """

    def __init__(
        self,
        settings: Settings,
        client: AsyncOpenAI | None = None,
        provider: str = "openai",
    ) -> None:
        self._model = settings.resolved_embedding_model
        self._label = provider
        self._client = client or _build_client(settings, provider)

    @property
    def model(self) -> str:
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = list(texts[start : start + _EMBED_BATCH_SIZE])
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed_batch([text])
        return vectors[0]

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(model=self._model, input=batch)
        except Exception as exc:
            raise _translate(exc, self._label) from exc

        if len(response.data) != len(batch):
            raise ProviderResponseError(
                f"{self._label} returned {len(response.data)} embeddings for {len(batch)} inputs"
            )
        # The API documents but does not guarantee ordering, and a silent
        # misalignment here would attach every chunk to the wrong vector.
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    async def aclose(self) -> None:
        await self._client.close()
