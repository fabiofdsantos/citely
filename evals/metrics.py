"""Evaluation metrics.

Four numbers, each answering a question a reviewer would actually ask:

* **retrieval hit rate** — did the right source come back at all? Everything
  downstream is capped by this, so a low answer rate with a low hit rate is a
  retrieval problem, not a model problem.
* **behaviour accuracy** — did the system answer when it should and refuse when
  it should? Reported split into the two directions, because they fail for
  opposite reasons and have opposite costs.
* **citation precision** — of the citations the model claimed, how many survived
  verification? This is the direct measurement of fabrication attempts.
* **groundedness** — did every returned answer carry at least one verified
  citation? This must be 1.0 by construction; if it ever isn't, the guardrail
  is broken, and that is worth failing CI over.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from citely.models import Answer
from citely.rag.answerer import AnswerTrace
from evals.dataset import Case


class CaseResult(BaseModel):
    """What the system did for one case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case: Case
    answer: Answer
    trace: AnswerTrace

    @property
    def behaved_as_expected(self) -> bool:
        return self.answer.refused == (self.case.expect == "refuse")

    @property
    def retrieved_evidence(self) -> bool | None:
        """Whether the expected source was retrieved, or None if none declared."""
        if self.case.evidence is None:
            return None
        return any(self.case.evidence in uri for uri in self.trace.source_uris)

    @property
    def mentioned_expected_terms(self) -> bool | None:
        """Weak signal: does an answer mention the terms a correct one would?"""
        if not self.case.must_mention or self.answer.refused:
            return None
        text = self.answer.text.casefold()
        return all(term.casefold() in text for term in self.case.must_mention)

    @property
    def is_grounded(self) -> bool:
        """A non-refusal must carry at least one verified citation."""
        return self.answer.refused or bool(self.answer.citations)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


class Report(BaseModel):
    """Aggregate metrics over a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    results: list[CaseResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def retrieval_hit_rate(self) -> float:
        checked = [r for r in self.results if r.retrieved_evidence is not None]
        return _ratio(sum(1 for r in checked if r.retrieved_evidence), len(checked))

    @property
    def answer_accuracy(self) -> float:
        """Of the cases that should be answered, how many were?"""
        expected = [r for r in self.results if r.case.expect == "answer"]
        return _ratio(sum(1 for r in expected if r.behaved_as_expected), len(expected))

    @property
    def refusal_accuracy(self) -> float:
        """Of the cases that should be refused, how many were?"""
        expected = [r for r in self.results if r.case.expect == "refuse"]
        return _ratio(sum(1 for r in expected if r.behaved_as_expected), len(expected))

    @property
    def citation_precision(self) -> float:
        claimed = sum(r.trace.claimed_citations for r in self.results)
        verified = sum(r.trace.verified_citations for r in self.results)
        return _ratio(verified, claimed)

    @property
    def groundedness(self) -> float:
        return _ratio(sum(1 for r in self.results if r.is_grounded), self.total)

    @property
    def keyword_coverage(self) -> float:
        checked = [r for r in self.results if r.mentioned_expected_terms is not None]
        return _ratio(sum(1 for r in checked if r.mentioned_expected_terms), len(checked))

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.behaved_as_expected or not r.is_grounded]

    def as_dict(self) -> dict[str, float]:
        return {
            "retrieval_hit_rate": self.retrieval_hit_rate,
            "answer_accuracy": self.answer_accuracy,
            "refusal_accuracy": self.refusal_accuracy,
            "citation_precision": self.citation_precision,
            "groundedness": self.groundedness,
            "keyword_coverage": self.keyword_coverage,
        }


class Thresholds(BaseModel):
    """Minimum acceptable scores. Falling below any of them fails the run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Groundedness is the only one at 1.0: it is enforced by code, not coaxed
    # out of a model, so anything less is a bug rather than a bad day.
    groundedness: float = 1.0
    retrieval_hit_rate: float = 0.8
    answer_accuracy: float = 0.8
    refusal_accuracy: float = 0.8
    citation_precision: float = 0.9
    keyword_coverage: float = 0.6

    def breaches(self, report: Report) -> list[str]:
        """Return one message per metric that fell short."""
        scores = report.as_dict()
        return [
            f"{name}: {scores[name]:.2f} < {minimum:.2f}"
            for name, minimum in self.model_dump().items()
            if scores[name] < minimum
        ]


def summarise(results: Sequence[CaseResult]) -> Report:
    return Report(results=list(results))
