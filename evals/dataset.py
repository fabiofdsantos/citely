"""The golden dataset: schema and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

GOLDEN_PATH = Path(__file__).parent / "golden.yaml"
CORPUS_PATH = Path(__file__).parent / "corpus"

Expectation = Literal["answer", "refuse"]


class Case(BaseModel):
    """One evaluation case.

    Expectations are behavioural. Asserting on exact answer text would measure
    the model's phrasing rather than whether the system stayed grounded, and
    would break on every model upgrade.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expect: Expectation
    evidence: str | None = None
    must_mention: list[str] = Field(default_factory=list)
    must_not_mention: list[str] = Field(
        default_factory=list,
        description=(
            "Substrings that must never appear in the answer. Used for canary "
            "phrases planted by injection payloads in the corpus."
        ),
    )
    because: str | None = None
    notes: str | None = None


class GoldenSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: list[Case] = Field(min_length=1)

    @property
    def expected_answers(self) -> list[Case]:
        return [c for c in self.cases if c.expect == "answer"]

    @property
    def expected_refusals(self) -> list[Case]:
        return [c for c in self.cases if c.expect == "refuse"]


def load_golden(path: Path = GOLDEN_PATH) -> GoldenSet:
    """Load and validate the golden set."""
    # safe_load, never load: the dataset is a data file, not a code path.
    return GoldenSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
