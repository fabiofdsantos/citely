"""Tests for the evaluation harness itself.

An eval suite that cannot fail is worthless. The important test here runs the
whole harness against a deliberately fabricating model and asserts the run goes
red — that is what makes a green run mean something.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from evals.dataset import CORPUS_PATH, Case, Expectation, load_golden
from evals.metrics import CaseResult, Thresholds, summarise
from evals.offline import ExtractiveLLM, HashingEmbedder
from evals.run import main, run_case

from citely.ingest.chunking import chunk_document
from citely.ingest.loaders import load_corpus
from citely.models import Answer, Citation, EmbeddedChunk
from citely.providers.base import Completion, TokenUsage
from citely.rag.answerer import Answerer, AnswerTrace
from citely.rag.retriever import VectorRetriever
from tests.fakes import InMemoryVectorStore


class FabricatingLLM:
    """Cites sources that exist, with quotes that do not.

    This is the failure mode the whole system is built against: fluent, well
    formed, correctly structured, and unsupported.
    """

    model = "fabricating-v1"

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        payload = {
            "answer": "The Regulation requires annual third-party audits of all systems [1].",
            "citations": [
                {"source": 1, "quote": "The Regulation requires annual third-party audits."}
            ],
            "insufficient_context": False,
            "reason": None,
        }
        return Completion(
            text=json.dumps(payload),
            model=self.model,
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )

    async def aclose(self) -> None:
        return None


async def build_answerer(llm: Any) -> Answerer:
    embedder, store = HashingEmbedder(), InMemoryVectorStore()
    for document in load_corpus(CORPUS_PATH):
        chunks = chunk_document(document, chunk_size=600, overlap=80)
        vectors = await embedder.embed_documents([c.text for c in chunks])
        await store.upsert(
            [
                EmbeddedChunk(chunk=chunk, embedding=vector, embedding_model=embedder.model)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
        )
    return Answerer(VectorRetriever(embedder, store, top_k=4), llm)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


class TestGoldenSet:
    def test_dataset_loads_and_validates(self) -> None:
        golden = load_golden()
        assert len(golden.cases) >= 8

    def test_case_ids_are_unique(self) -> None:
        ids = [c.id for c in load_golden().cases]
        assert len(ids) == len(set(ids))

    def test_both_behaviours_are_represented(self) -> None:
        """A set of only-answer cases would never test the refusal path."""
        golden = load_golden()
        assert golden.expected_answers
        assert golden.expected_refusals

    def test_refusal_cases_explain_themselves(self) -> None:
        assert all(c.because for c in load_golden().expected_refusals)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def case(expect: Expectation = "answer", evidence: str | None = None) -> Case:
    return Case(id="c", question="q?", expect=expect, evidence=evidence, because="r")


def result(
    *,
    expect: Expectation = "answer",
    refused: bool = False,
    claimed: int = 1,
    verified: int = 1,
    uris: list[str] | None = None,
    evidence: str | None = None,
) -> CaseResult:
    answer = (
        Answer(query="q?", text="refused", refused=True, refusal_reason="because")
        if refused
        else Answer(
            query="q?",
            text="answer [1]",
            citations=[Citation(chunk_id="c1", quote="a quote here", source_uri="corpus/a.md")],
        )
    )
    return CaseResult(
        case=case(expect, evidence),
        answer=answer,
        trace=AnswerTrace(
            claimed_citations=claimed,
            verified_citations=verified,
            source_uris=uris or ["corpus/a.md"],
        ),
    )


class TestMetrics:
    def test_perfect_run_scores_one(self) -> None:
        report = summarise([result(), result(expect="refuse", refused=True)])
        assert report.answer_accuracy == 1.0
        assert report.refusal_accuracy == 1.0
        assert report.groundedness == 1.0

    def test_missed_refusal_lowers_refusal_accuracy(self) -> None:
        """Answering when it should have refused is the expensive mistake."""
        report = summarise([result(expect="refuse", refused=False)])
        assert report.refusal_accuracy == 0.0
        assert report.failures

    def test_over_refusal_lowers_answer_accuracy(self) -> None:
        report = summarise([result(expect="answer", refused=True)])
        assert report.answer_accuracy == 0.0

    def test_rejected_citations_lower_precision(self) -> None:
        report = summarise([result(claimed=4, verified=1)])
        assert report.citation_precision == 0.25

    def test_hit_rate_counts_only_cases_declaring_evidence(self) -> None:
        report = summarise(
            [
                result(evidence="a.md", uris=["corpus/a.md"]),
                result(evidence="missing.md", uris=["corpus/a.md"]),
                result(evidence=None),
            ]
        )
        assert report.retrieval_hit_rate == 0.5

    def test_empty_denominators_do_not_divide_by_zero(self) -> None:
        assert summarise([]).as_dict()["groundedness"] == 1.0


class TestThresholds:
    def test_passing_report_has_no_breaches(self) -> None:
        report = summarise([result(), result(expect="refuse", refused=True)])
        assert Thresholds().breaches(report) == []

    def test_breaches_name_the_metric_and_the_numbers(self) -> None:
        report = summarise([result(claimed=10, verified=1)])
        [breach] = [b for b in Thresholds().breaches(report) if b.startswith("citation_precision")]
        assert "0.10" in breach


# --------------------------------------------------------------------------
# The harness end to end
# --------------------------------------------------------------------------


class TestHarness:
    async def test_offline_stubs_pass_the_guardrail_thresholds(self) -> None:
        answerer = await build_answerer(ExtractiveLLM())
        golden = load_golden()

        report = summarise([await run_case(answerer, c) for c in golden.cases])

        assert report.groundedness == 1.0
        assert report.citation_precision == 1.0

    async def test_a_fabricating_model_fails_the_run(self) -> None:
        """The load-bearing test: green must be earned, not guaranteed."""
        answerer = await build_answerer(FabricatingLLM())
        golden = load_golden()

        report = summarise([await run_case(answerer, c) for c in golden.cases])

        assert report.citation_precision == 0.0
        assert report.answer_accuracy == 0.0, "fabricated answers must not count as answers"
        assert Thresholds().breaches(report)

    async def test_fabricated_answers_are_converted_to_refusals(self) -> None:
        answerer = await build_answerer(FabricatingLLM())

        results = [await run_case(answerer, c) for c in load_golden().cases]

        assert all(r.answer.refused for r in results)
        assert all(r.trace.fabricated_citations > 0 for r in results if r.trace.claimed_citations)


def test_runner_exits_zero_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--offline"]) == 0
    assert "All thresholds met" in capsys.readouterr().out
