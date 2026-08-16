"""Provider interfaces and the wire types they exchange.

Two protocols rather than one: Anthropic ships no embeddings API, so a single
``Provider`` interface would force every implementation to lie about half its
surface. Splitting them also lets the chat and embedding backends be chosen
independently, which is the common production setup.

``Protocol`` (structural typing) is used instead of an abstract base class so
implementations need no shared parent and test fakes are plain objects — the
closest Python gets to a TypeScript ``interface``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["user", "assistant"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Message(_Frozen):
    """One turn of a conversation.

    System prompts are passed separately to ``complete`` rather than as a
    message, because Anthropic models them as a top-level parameter and OpenAI
    as a message. Keeping them out of this list means neither backend has to
    reach in and rewrite the caller's history.
    """

    role: Role
    content: str = Field(min_length=1)


class TokenUsage(_Frozen):
    """Token counts for one call, for cost tracking and context budgeting."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Completion(_Frozen):
    """A chat completion, normalised across providers."""

    text: str
    model: str = Field(min_length=1)
    usage: TokenUsage
    stop_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """Whether generation stopped at the token limit rather than finishing.

        Worth checking before parsing structured output: a truncated response is
        the usual cause of "the model returned invalid JSON".
        """
        return self.stop_reason in {"max_tokens", "length"}


@runtime_checkable
class LLMProvider(Protocol):
    """A chat-completion backend."""

    @property
    def model(self) -> str:
        """The model id this provider was configured with."""
        ...

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        """Generate a completion.

        Raises:
            ProviderError: on any transport, auth, rate-limit or shape failure.
        """
        ...

    async def aclose(self) -> None:
        """Release the underlying HTTP connections."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """An embedding backend.

    Vector dimensionality is deliberately not part of this interface: it is a
    property of the model's output, discovered from the first vector and then
    recorded by the store. Declaring it up front invites a hardcoded constant
    that quietly disagrees with reality.
    """

    @property
    def model(self) -> str:
        """The model id this provider was configured with."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents, returning one vector per input, in order."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Separate from :meth:`embed_documents` because some models require an
        asymmetric prefix or a different task type for queries.
        """
        ...

    async def aclose(self) -> None:
        """Release the underlying HTTP connections."""
        ...
