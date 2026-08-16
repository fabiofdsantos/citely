import pytest
from pydantic import ValidationError

from citely.config import Settings, get_settings
from citely.errors import ConfigurationError


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


def test_defaults_resolve_per_provider(settings: Settings) -> None:
    assert settings.llm_provider == "anthropic"
    assert settings.resolved_llm_model == "claude-sonnet-5"
    assert settings.resolved_embedding_model == "text-embedding-3-small"


def test_missing_llm_credential_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        Settings(llm_provider="anthropic")


def test_all_missing_credentials_are_reported_together() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings(llm_provider="anthropic")
    message = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "OPENAI_API_KEY" in message


def test_explicit_model_overrides_provider_default(keys: None) -> None:
    assert Settings(llm_model="gpt-4o").resolved_llm_model == "gpt-4o"


def test_api_key_read_from_unprefixed_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing shells export OPENAI_API_KEY, not CITELY_OPENAI_API_KEY."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(llm_provider="openai")
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test"


def test_secret_is_not_exposed_by_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-log-me")
    settings = Settings(llm_provider="openai")
    assert "sk-do-not-log-me" not in repr(settings)


def test_overlap_must_be_smaller_than_chunk_size(keys: None) -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(chunk_size=500, chunk_overlap=500)


def test_pgvector_requires_a_dsn(keys: None) -> None:
    with pytest.raises(ValidationError, match="PGVECTOR_DSN"):
        Settings(vector_store="pgvector")


def test_openai_base_url_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any OpenAI-compatible endpoint is reachable without a code change."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CITELY_OPENAI_BASE_URL", "https://gateway.internal/v1")
    assert Settings(llm_provider="openai").openai_base_url == "https://gateway.internal/v1"


def test_get_settings_is_cached(keys: None) -> None:
    assert get_settings() is get_settings()


def test_get_settings_raises_configuration_error() -> None:
    """Callers catch one exception type; pydantic's ValidationError does not leak."""
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        get_settings()
