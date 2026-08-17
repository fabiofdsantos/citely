"""CLI tests.

These run the real command end to end — real Chroma on a temp path, a stub
embedder patched in — so they catch wiring failures that unit tests cannot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from citely.cli import app
from tests.fakes import FakeEmbedder, FakeLLM

runner = CliRunner()


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A configured corpus plus an offline embedder, wired through env vars."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "act.md").write_text(
        "# EU AI Act\n\n" + "\n\n".join(f"Article {i} says something." for i in range(8)),
        encoding="utf-8",
    )

    monkeypatch.setenv("CITELY_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("CITELY_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("CITELY_CORPUS_PATH", str(corpus))
    monkeypatch.setenv("CITELY_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CITELY_CHUNK_SIZE", "120")
    monkeypatch.setenv("CITELY_CHUNK_OVERLAP", "0")

    # Patch at the point of use: the CLI resolves the provider through the
    # registry, so replacing that keeps the rest of the wiring real.
    monkeypatch.setattr("citely.cli.build_embedding_provider", lambda _settings: FakeEmbedder())
    return corpus


def test_ingest_reports_what_it_indexed(workspace: Path) -> None:
    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 0, result.output
    assert "act.md" in result.output
    assert "embedded" in result.output


def test_reingest_is_a_no_op(workspace: Path) -> None:
    runner.invoke(app, ["ingest"])

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 0, result.output
    assert "0 embedded" in result.output
    assert "unchanged" in result.output


def test_ingest_accepts_an_explicit_path(workspace: Path) -> None:
    result = runner.invoke(app, ["ingest", str(workspace / "act.md")])

    assert result.exit_code == 0, result.output
    assert "act.md" in result.output


def test_missing_corpus_exits_nonzero(workspace: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["ingest", str(tmp_path / "absent")])

    assert result.exit_code == 1
    assert "ingestion_error" in result.output


def test_status_reports_the_index(workspace: Path) -> None:
    runner.invoke(app, ["ingest"])

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "fake-embed" in result.output
    assert "chroma" in result.output


def test_configuration_errors_exit_with_code_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad config is distinguishable from a failed run by exit code alone."""
    monkeypatch.setenv("CITELY_LLM_PROVIDER", "anthropic")

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 2
    assert "configuration error" in result.output


def test_query_prints_answer_and_sources(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # One chunk for the whole file, so the source the model cites is knowable
    # from the test rather than dependent on the stub embedder's ranking.
    monkeypatch.setenv("CITELY_CHUNK_SIZE", "2000")
    runner.invoke(app, ["ingest"])
    monkeypatch.setattr(
        "citely.cli.build_llm_provider",
        lambda _settings: FakeLLM(
            json.dumps(
                {
                    "answer": "Article 0 says something [1].",
                    "citations": [{"source": 1, "quote": "Article 0 says something."}],
                    "insufficient_context": False,
                    "reason": None,
                }
            )
        ),
    )

    result = runner.invoke(app, ["query", "what does article 0 say?"])

    assert result.exit_code == 0, result.output
    assert "Article 0 says something" in result.output
    assert "Sources:" in result.output


def test_query_prints_a_refusal_without_sources(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["ingest"])
    monkeypatch.setattr(
        "citely.cli.build_llm_provider",
        lambda _settings: FakeLLM(
            json.dumps(
                {
                    "answer": "",
                    "citations": [],
                    "insufficient_context": True,
                    "reason": "the corpus does not cover this",
                }
            )
        ),
    )

    # No proper nouns, so the scope check passes it through to the model, whose
    # own "insufficient context" refusal is what this test covers.
    result = runner.invoke(app, ["query", "does the act require annual audits?"])

    assert result.exit_code == 0, result.output
    assert "does not cover" in result.output
    assert "Sources:" not in result.output


def test_empty_question_exits_with_code_two(workspace: Path) -> None:
    result = runner.invoke(app, ["query", "  "])

    assert result.exit_code == 2
    assert "invalid question" in result.output
