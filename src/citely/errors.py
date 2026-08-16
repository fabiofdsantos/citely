"""Exception hierarchy.

One base class so callers can catch everything citely raises, and a stable
machine-readable ``code`` on each so the API layer can map errors to responses
without a pile of ``isinstance`` checks.
"""


class CitelyError(Exception):
    """Base class for every error raised by citely."""

    code = "internal_error"


class ConfigurationError(CitelyError):
    """Settings are missing, contradictory, or unusable."""

    code = "configuration_error"


class ProviderError(CitelyError):
    """An LLM or embedding provider call failed."""

    code = "provider_error"


class ProviderAuthError(ProviderError):
    """The provider rejected our credentials."""

    code = "provider_auth_error"


class ProviderRateLimitError(ProviderError):
    """The provider throttled us."""

    code = "provider_rate_limit"


class ProviderTimeoutError(ProviderError):
    """The provider did not respond in time."""

    code = "provider_timeout"


class ProviderResponseError(ProviderError):
    """The provider responded, but not in a shape we can use."""

    code = "provider_response_error"


class StoreError(CitelyError):
    """The vector store failed."""

    code = "store_error"


class EmbeddingMismatchError(StoreError):
    """The collection was built with a different embedding model or dimensionality.

    Silently mixing embedding spaces produces retrieval that looks fine and is
    meaningless, so this is always fatal rather than a warning.
    """

    code = "embedding_mismatch"


class IngestionError(CitelyError):
    """A document could not be loaded, parsed, or chunked."""

    code = "ingestion_error"


class RetrievalError(CitelyError):
    """Retrieval failed before generation could start."""

    code = "retrieval_error"
