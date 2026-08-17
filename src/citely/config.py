"""Application settings.

Everything is configured through environment variables (12-factor), validated
once at startup by pydantic-settings. Invalid configuration fails immediately
with a readable message rather than at the first API call.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from citely.errors import ConfigurationError

LLMProviderName = Literal["openai", "anthropic", "ollama"]
EmbeddingProviderName = Literal["openai", "ollama"]
VectorStoreName = Literal["chroma", "pgvector"]

#: Chat models used when ``CITELY_LLM_MODEL`` is unset.
DEFAULT_LLM_MODELS: Final[dict[LLMProviderName, str]] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "ollama": "llama3.1:8b",
}

#: Embedding models used when ``CITELY_EMBEDDING_MODEL`` is unset.
DEFAULT_EMBEDDING_MODELS: Final[dict[EmbeddingProviderName, str]] = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}

#: Providers that speak the OpenAI wire format and therefore share one client
#: implementation, differing only in base URL and credentials.
OPENAI_COMPATIBLE: Final[frozenset[str]] = frozenset({"openai", "ollama"})


class Settings(BaseSettings):
    """Validated runtime configuration.

    Read via :func:`get_settings` rather than instantiated directly, so the
    whole process shares one validated instance.
    """

    model_config = SettingsConfigDict(
        env_prefix="CITELY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Surface every problem at once instead of one per run.
        validate_default=True,
    )

    # -- Providers ---------------------------------------------------------
    llm_provider: LLMProviderName = "anthropic"
    embedding_provider: EmbeddingProviderName = "openai"

    llm_model: str | None = Field(
        default=None,
        description="Chat model id. Defaults per provider; see DEFAULT_LLM_MODELS.",
    )
    embedding_model: str | None = Field(
        default=None,
        description="Embedding model id. Defaults per provider.",
    )

    # Credentials are also read from the conventional unprefixed names so that
    # existing shells and CI secrets work untouched.
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CITELY_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CITELY_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )

    openai_base_url: str | None = Field(
        default=None,
        description=(
            "Override the OpenAI API endpoint. Any server speaking the same wire "
            "format (Azure OpenAI, a gateway, a proxy) works without code changes."
        ),
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Ollama's OpenAI-compatible endpoint. Note the /v1 suffix.",
    )

    request_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)

    # -- Storage -----------------------------------------------------------
    vector_store: VectorStoreName = "chroma"
    collection_name: str = "citely"
    chroma_path: Path = Path("./data/chroma")
    pgvector_dsn: SecretStr | None = None

    # -- Retrieval ---------------------------------------------------------
    top_k: int = Field(default=6, ge=1, le=50)
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Chunks scoring below this are discarded before generation.",
    )
    scope_check: bool = Field(
        default=True,
        description=(
            "Refuse before generating when the question names something the "
            "retrieved sources never mention. Guards against grounded answers "
            "to questions the corpus does not actually cover."
        ),
    )
    max_context_tokens: int = Field(
        default=6000,
        ge=256,
        description="Token budget for retrieved context; small local models need small budgets.",
    )

    # -- Ingestion ---------------------------------------------------------
    corpus_path: Path = Path("./data/corpus")
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0)

    # -- Observability -----------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # -- Normalisation -----------------------------------------------------
    @field_validator("openai_api_key", "anthropic_api_key", "pgvector_dsn", mode="before")
    @classmethod
    def _empty_secret_is_absent(cls, value: object) -> object:
        """Treat a blank credential as missing rather than as a value.

        ``.env.example`` ships keys as ``OPENAI_API_KEY=`` for the user to fill
        in. Without this, copying that file and filling in only one provider's
        key leaves the other as an empty string — which passes the "is it set?"
        check and then fails deep inside a provider SDK at startup, long after
        the configuration error could have been reported clearly.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # -- Derived values ----------------------------------------------------
    @property
    def resolved_llm_model(self) -> str:
        """The chat model to use, falling back to the provider's default."""
        return self.llm_model or DEFAULT_LLM_MODELS[self.llm_provider]

    @property
    def resolved_embedding_model(self) -> str:
        """The embedding model to use, falling back to the provider's default."""
        return self.embedding_model or DEFAULT_EMBEDDING_MODELS[self.embedding_provider]

    def base_url_for(self, provider: str) -> str | None:
        """Return the endpoint for an OpenAI-compatible provider, if it has one."""
        return self.ollama_base_url if provider == "ollama" else self.openai_base_url

    # -- Cross-field validation -------------------------------------------
    @model_validator(mode="after")
    def _check_chunking(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            msg = (
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); otherwise chunking never advances."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_credentials(self) -> Self:
        """Fail at startup when a selected provider has no usable credential.

        Ollama is deliberately exempt: a local runtime needs no key, which is
        what makes the zero-credential quickstart possible.
        """
        missing: list[str] = []
        if self.llm_provider == "openai" and self.openai_api_key is None:
            missing.append("OPENAI_API_KEY (required by CITELY_LLM_PROVIDER=openai)")
        if self.llm_provider == "anthropic" and self.anthropic_api_key is None:
            missing.append("ANTHROPIC_API_KEY (required by CITELY_LLM_PROVIDER=anthropic)")
        if self.embedding_provider == "openai" and self.openai_api_key is None:
            missing.append("OPENAI_API_KEY (required by CITELY_EMBEDDING_PROVIDER=openai)")

        if missing:
            raise ValueError("missing credentials: " + "; ".join(sorted(set(missing))))
        return self

    @model_validator(mode="after")
    def _check_store(self) -> Self:
        if self.vector_store == "pgvector" and self.pgvector_dsn is None:
            msg = "CITELY_PGVECTOR_DSN is required when CITELY_VECTOR_STORE=pgvector"
            raise ValueError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, validating them on first access.

    Cached so that importing modules can call this freely; call
    ``get_settings.cache_clear()`` in tests that manipulate the environment.
    """
    try:
        return Settings()
    except ValueError as exc:  # pydantic ValidationError subclasses ValueError
        raise ConfigurationError(str(exc)) from exc
