"""Loading documents off disk.

Kept deliberately small: a corpus is a directory of text files. Adding PDF or
HTML support means adding a parser here and nothing else, because everything
downstream consumes :class:`~citely.models.RawDocument`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from citely.errors import IngestionError
from citely.models import RawDocument

#: Extensions treated as plain text. Anything else is skipped with a warning
#: rather than silently ingested as mojibake.
TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".rst"})


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"{path} is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise IngestionError(f"cannot read {path}: {exc}") from exc


def _title_from(path: Path, text: str) -> str:
    """Prefer a leading Markdown heading, fall back to the filename."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped:
            break
    return path.stem.replace("_", " ").replace("-", " ")


def load_file(path: Path) -> RawDocument:
    """Load a single file into a document."""
    text = _read(path)
    if not text.strip():
        raise IngestionError(f"{path} is empty")

    return RawDocument(
        # Relative where possible so that ids stay stable across machines and
        # containers; an absolute path would change every id on redeploy.
        source_uri=_relative_uri(path),
        text=text,
        title=_title_from(path, text),
        metadata={"suffix": path.suffix},
    )


def _relative_uri(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_corpus(root: Path) -> Iterator[RawDocument]:
    """Yield every supported document under ``root``, sorted for determinism.

    Sorting matters: it makes ingestion order reproducible, which in turn makes
    ingestion reports diffable between runs.
    """
    if not root.exists():
        raise IngestionError(f"corpus path does not exist: {root}")

    if root.is_file():
        yield load_file(root)
        return

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
    if not files:
        suffixes = ", ".join(sorted(TEXT_SUFFIXES))
        raise IngestionError(f"no documents found under {root} (looking for: {suffixes})")

    for path in files:
        yield load_file(path)
