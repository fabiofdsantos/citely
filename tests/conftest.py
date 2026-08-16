"""Shared fixtures.

The one job here is hermetic settings: a developer's real ``.env`` and exported
API keys must never leak into a test run, or the suite passes on their machine
and fails in CI.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from citely.config import Settings, get_settings

_LEAKY_PREFIXES = ("CITELY_",)
_LEAKY_NAMES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Strip citely-related env vars and run from a directory with no ``.env``."""
    import os

    for name in list(os.environ):
        if name.startswith(_LEAKY_PREFIXES) or name in _LEAKY_NAMES:
            monkeypatch.delenv(name, raising=False)

    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Valid settings backed by dummy credentials, usable by any test."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    return Settings()
