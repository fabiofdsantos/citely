"""Command-line entrypoint. Subcommands land here as the pipeline is built."""

import typer

from citely import __version__

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
    `citely version` would be parsed as an unexpected argument. Keeping it means
    subcommands stay subcommands as we add `ingest` and `query`.
    """


@app.command()
def version() -> None:
    """Print the installed citely version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
