"""Command-line entrypoint.

The CLI exists so the tool is usable without running the server, and so the
ingestion path can be exercised in CI without HTTP in the way.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from citely import __version__
from citely.config import Settings, get_settings
from citely.errors import CitelyError
from citely.ingest.pipeline import IngestReport, ingest_path
from citely.models import Answer
from citely.providers.registry import build_embedding_provider, build_llm_provider
from citely.rag.answerer import Answerer
from citely.rag.guardrails import InvalidQueryError
from citely.rag.retriever import VectorRetriever
from citely.stores.registry import build_vector_store

app = typer.Typer(
    name="citely",
    help="Ask questions over a document corpus and get grounded, cited answers.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Root callback.

    Without this, Typer collapses a single-command app into a bare command and
    `citely version` would be parsed as an unexpected argument.
    """


def _load_settings() -> Settings:
    """Resolve settings, turning configuration problems into clean CLI errors."""
    try:
        return get_settings()
    except CitelyError as exc:
        typer.secho(f"configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def version() -> None:
    """Print the installed citely version."""
    typer.echo(__version__)


@app.command()
def ingest(
    path: Annotated[
        Path | None,
        typer.Argument(help="File or directory to ingest. Defaults to CITELY_CORPUS_PATH."),
    ] = None,
) -> None:
    """Load, chunk, embed and index a corpus.

    Safe to re-run: unchanged chunks are never re-embedded, and content removed
    from the source is removed from the index.
    """
    settings = _load_settings()
    corpus = path or settings.corpus_path

    try:
        report = asyncio.run(_run_ingest(settings, corpus))
    except CitelyError as exc:
        typer.secho(f"{exc.code}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    _print_report(report, corpus)


async def _run_ingest(settings: Settings, corpus: Path) -> IngestReport:
    embedder = build_embedding_provider(settings)
    store = build_vector_store(settings)
    try:
        return await ingest_path(
            corpus,
            embedder=embedder,
            store=store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    finally:
        await embedder.aclose()


def _print_report(report: IngestReport, corpus: Path) -> None:
    for document in report.documents:
        state = "unchanged" if document.unchanged else "updated"
        typer.echo(
            f"  {document.source_uri}: {document.total_chunks} chunks "
            f"({document.embedded} embedded, {document.skipped} skipped, "
            f"{document.deleted} deleted) [{state}]"
        )

    summary = (
        f"{len(report.documents)} documents from {corpus}: "
        f"{report.embedded} embedded, {report.skipped} skipped, {report.deleted} deleted"
    )
    typer.secho(summary, fg=typer.colors.GREEN if report.embedded else typer.colors.BLUE)


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="The question to answer from the corpus.")],
    k: Annotated[int | None, typer.Option("--top-k", help="Chunks to retrieve.")] = None,
) -> None:
    """Answer a question from the indexed corpus, with citations."""
    settings = _load_settings()

    try:
        answer = asyncio.run(_run_query(settings, question, k))
    except InvalidQueryError as exc:
        typer.secho(f"invalid question: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    except CitelyError as exc:
        typer.secho(f"{exc.code}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if answer.refused:
        typer.secho(answer.text, fg=typer.colors.YELLOW)
        return

    typer.echo(answer.text)
    typer.echo("\nSources:")
    for number, citation in enumerate(answer.citations, start=1):
        typer.echo(f'  [{number}] {citation.title or citation.source_uri} — "{citation.quote}"')
        typer.echo(f"      {citation.source_uri}#{citation.chunk_id}")


async def _run_query(settings: Settings, question: str, k: int | None) -> Answer:
    embedder = build_embedding_provider(settings)
    llm = build_llm_provider(settings)
    try:
        retriever = VectorRetriever(
            embedder,
            build_vector_store(settings),
            top_k=settings.top_k,
            min_score=settings.min_score,
        )
        answerer = Answerer(
            retriever,
            llm,
            max_context_tokens=settings.max_context_tokens,
            scope_check=settings.scope_check,
            scope_ignore_terms=settings.scope_ignored_terms,
        )
        return await answerer.answer(question, k=k)
    finally:
        await embedder.aclose()
        await llm.aclose()


@app.command()
def status() -> None:
    """Show what is currently indexed."""
    settings = _load_settings()
    try:
        health = asyncio.run(build_vector_store(settings).health())
    except CitelyError as exc:
        typer.secho(f"{exc.code}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"backend:         {health.backend}")
    typer.echo(f"collection:      {health.collection}")
    typer.echo(f"chunks:          {health.chunk_count}")
    typer.echo(f"embedding model: {health.embedding_model or '-'}")
    typer.echo(f"dimensions:      {health.dimensions or '-'}")


if __name__ == "__main__":  # pragma: no cover
    app()
