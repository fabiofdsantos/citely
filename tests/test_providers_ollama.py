"""Ollama-specific behaviour.

Ollama reuses the OpenAI client wholesale — these tests pin the three things
that differ: no credential, a different base URL, and the failure mode of a
local runtime that is simply not running.
"""

from __future__ import annotations

import openai
import pytest

from citely.config import Settings
from citely.errors import ProviderError
from citely.providers.openai import OpenAIChatProvider, OpenAIEmbeddingProvider
from citely.providers.registry import build_embedding_provider, build_llm_provider
from tests.fakes import FakeOpenAIClient, as_openai, http_request


@pytest.fixture
def ollama_settings() -> Settings:
    return Settings(llm_provider="ollama", embedding_provider="ollama")


def test_ollama_needs_no_credentials(ollama_settings: Settings) -> None:
    """The zero-key quickstart only works if a local provider validates cleanly."""
    assert ollama_settings.resolved_llm_model == "llama3.1:8b"
    assert ollama_settings.resolved_embedding_model == "nomic-embed-text"


def test_base_url_is_selected_per_provider(ollama_settings: Settings) -> None:
    assert ollama_settings.base_url_for("ollama") == "http://localhost:11434/v1"
    assert ollama_settings.base_url_for("openai") is None


def test_client_is_built_without_an_api_key(ollama_settings: Settings) -> None:
    """A real client, not a fake: constructing it must not demand a credential."""
    provider = OpenAIChatProvider(ollama_settings, provider="ollama")
    assert provider.model == "llama3.1:8b"


def test_registry_routes_ollama_to_the_openai_client(ollama_settings: Settings) -> None:
    assert isinstance(build_llm_provider(ollama_settings), OpenAIChatProvider)
    assert isinstance(build_embedding_provider(ollama_settings), OpenAIEmbeddingProvider)


async def test_unreachable_runtime_reports_the_base_url(ollama_settings: Settings) -> None:
    """ "Connection refused" is the first thing a user hits; say so in plain words."""
    unreachable = openai.APIConnectionError(request=http_request())
    client = FakeOpenAIClient(embedding_response=unreachable)
    provider = OpenAIEmbeddingProvider(ollama_settings, client=as_openai(client), provider="ollama")

    with pytest.raises(ProviderError, match="ollama is unreachable"):
        await provider.embed_query("what is a high-risk system?")
