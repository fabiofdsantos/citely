"""Config-driven construction of providers.

The rest of the codebase depends on the protocols in ``base`` and never imports
a concrete backend; this module is the single place where a config string
becomes an object. Adding a provider means editing one table here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from citely.errors import ConfigurationError
from citely.providers.base import EmbeddingProvider, LLMProvider

if TYPE_CHECKING:
    from citely.config import Settings


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Construct the chat provider named by ``CITELY_LLM_PROVIDER``."""
    # Imported lazily so that selecting one provider does not pay the import
    # cost of the other's SDK.
    match settings.llm_provider:
        case "anthropic":
            from citely.providers.anthropic import AnthropicChatProvider

            return AnthropicChatProvider(settings)
        case "openai" | "ollama" as provider:
            from citely.providers.openai import OpenAIChatProvider

            return OpenAIChatProvider(settings, provider=provider)
        case unknown:  # pragma: no cover - Literal typing makes this unreachable
            raise ConfigurationError(f"unknown LLM provider: {unknown!r}")


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Construct the embedding provider named by ``CITELY_EMBEDDING_PROVIDER``."""
    match settings.embedding_provider:
        case "openai" | "ollama" as provider:
            from citely.providers.openai import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(settings, provider=provider)
        case unknown:  # pragma: no cover - Literal typing makes this unreachable
            raise ConfigurationError(f"unknown embedding provider: {unknown!r}")
