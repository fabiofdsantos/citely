"""Evaluation runner: ``make eval``.

Ingests the evaluation corpus into a throwaway index, runs every golden case,
prints a report, and exits non-zero when a metric falls below its threshold.

Two modes:

``--offline`` (the CI default)
    Uses the deterministic stubs in :mod:`evals.offline`. No API key, no
    network. Only the guardrail metrics are gated, because the stub's answer
    quality is not the thing under test — the harness and the verification path
    are.

``--live``
    Uses the configured providers and gates on the full threshold set. This is
    the number worth quoting: it measures the real system.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from citely.config import Settings, get_settings
from citely.errors import CitelyError, ConfigurationError
from citely.ingest.pipeline import ingest_path
from citely.models import Answer
from citely.providers.base import EmbeddingProvider, LLMProvider
from citely.providers.registry import build_embedding_provider, build_llm_provider
from citely.rag.answerer import Answerer, AnswerTrace
from citely.rag.guardrails import InvalidQueryError
from citely.rag.retriever import VectorRetriever
from citely.stores.base import VectorStore
from citely.stores.chroma import ChromaVectorStore
from evals.dataset import CORPUS_PATH, Case, load_golden
from evals.metrics import CaseResult, Report, Thresholds, summarise
from evals.offline import ExtractiveLLM, HashingEmbedder

#: Words this corpus is saturated with, which therefore say nothing about
#: whether a question is answerable from it. Corpus vocabulary lives in
#: configuration, never in the scope checker: a clinical or engineering corpus
#: would list entirely different words here.
EVAL_CORPUS_VOCABULARY = "ai,regulation,article,annex,union"

#: Offline runs gate only on what the code guarantees, never on answer quality.
OFFLINE_THRESHOLDS = Thresholds(
    groundedness=1.0,
    injection_resistance=1.0,
    citation_precision=1.0,
    retrieval_hit_rate=0.8,
    answer_accuracy=0.0,
    refusal_accuracy=0.0,
    keyword_coverage=0.0,
)


def _build_offline(
    chroma_path: Path,
) -> tuple[EmbeddingProvider, LLMProvider, VectorStore, Settings]:
    settings = Settings(
        llm_provider="ollama",  # never contacted; keeps validation credential-free
        embedding_provider="ollama",
        chroma_path=chroma_path,
        collection_name="evals",
        chunk_size=600,
        chunk_overlap=80,
        top_k=4,
        scope_ignore_terms=EVAL_CORPUS_VOCABULARY,
    )
    return HashingEmbedder(), ExtractiveLLM(), ChromaVectorStore(settings), settings


def _build_live(chroma_path: Path) -> tuple[EmbeddingProvider, LLMProvider, VectorStore, Settings]:
    # A throwaway collection: evaluation must never read or write the index a
    # user has built for real use.
    configured = get_settings()
    settings = configured.model_copy(
        update={
            "chroma_path": chroma_path,
            "collection_name": "evals",
            "scope_ignore_terms": EVAL_CORPUS_VOCABULARY,
        }
    )
    return (
        build_embedding_provider(settings),
        build_llm_provider(settings),
        ChromaVectorStore(settings),
        settings,
    )


async def run_case(answerer: Answerer, case: Case) -> CaseResult:
    try:
        answer, trace = await answerer.answer_with_trace(case.question)
    except InvalidQueryError as exc:
        # A rejected question is a refusal, not a crash: the guardrail fired
        # before retrieval, which for a hostile input is the desired outcome.
        answer = Answer(
            query=case.question,
            text=str(exc),
            refused=True,
            refusal_reason=str(exc),
        )
        trace = AnswerTrace()

    return CaseResult(case=case, answer=answer, trace=trace)


async def evaluate(*, offline: bool, corpus: Path) -> Report:
    with tempfile.TemporaryDirectory(prefix="citely-evals-") as tmp:
        build = _build_offline if offline else _build_live
        embedder, llm, store, settings = build(Path(tmp) / "chroma")

        try:
            await ingest_path(
                corpus,
                embedder=embedder,
                store=store,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            answerer = Answerer(
                VectorRetriever(
                    embedder, store, top_k=settings.top_k, min_score=settings.min_score
                ),
                llm,
                max_context_tokens=settings.max_context_tokens,
                scope_check=settings.scope_check,
                scope_ignore_terms=settings.scope_ignored_terms,
            )
            golden = load_golden()
            return summarise([await run_case(answerer, case) for case in golden.cases])
        finally:
            await embedder.aclose()
            await llm.aclose()


def _print_report(report: Report, *, mode: str) -> None:
    print(f"\ncitely evaluation ({mode})\n" + "=" * 60)

    for result in report.results:
        verdict = "PASS" if result.behaved_as_expected and result.is_grounded else "FAIL"
        actual = "refused" if result.answer.refused else "answered"
        print(f"  [{verdict}] {result.case.id}: expected {result.case.expect}, {actual}")
        if result.trace.rejections:
            for rejection in result.trace.rejections:
                print(f"           rejected citation: {rejection}")

    print("-" * 60)
    for name, value in report.as_dict().items():
        print(f"  {name:<22} {value:.2f}")
    print(f"  {'cases':<22} {report.total}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the citely evaluation suite.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic stubs; no API key needed (default).",
    )
    mode.add_argument(
        "--live", action="store_true", help="Use the configured providers and gate on all metrics."
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--json", action="store_true", help="Emit metrics as JSON.")
    args = parser.parse_args(argv)

    offline = not args.live
    try:
        report = asyncio.run(evaluate(offline=offline, corpus=args.corpus))
    except ConfigurationError as exc:
        # Missing keys is the expected first failure of `make eval-live`, not a
        # crash: say what is missing and how to supply it, without a traceback.
        print(f"\nconfiguration error: {exc}\n", file=sys.stderr)
        print(
            "Live evaluation needs provider credentials. Either:\n"
            "  cp .env.example .env   # then fill in the keys\n"
            "  export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...\n"
            "\nOr run without credentials against local models:\n"
            "  CITELY_LLM_PROVIDER=ollama CITELY_EMBEDDING_PROVIDER=ollama make eval-live\n"
            "\nOr run the offline harness check:\n"
            "  make eval",
            file=sys.stderr,
        )
        return 2
    except CitelyError as exc:
        print(f"\n{exc.code}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        _print_report(report, mode="offline stubs" if offline else "live providers")

    thresholds = OFFLINE_THRESHOLDS if offline else Thresholds()
    breaches = thresholds.breaches(report)
    if breaches:
        print("\nFAILED thresholds:")
        for breach in breaches:
            print(f"  {breach}")
        return 1

    print("\nAll thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
