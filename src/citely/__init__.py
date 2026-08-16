"""citely: provider-agnostic RAG with grounded, cited answers."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("citely")
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
