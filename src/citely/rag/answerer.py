"""Retrieval-augmented answering.

The pipeline is deliberately pessimistic. Every stage can end in a refusal, and
a refusal is a successful outcome — the failure mode this service exists to
prevent is a confident answer that no source supports.

    validate -> retrieve -> budget -> generate -> parse -> verify -> answer
                    |          |                    |         |
                    +----------+--------------------+---------+--> refusal
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from citely.errors import ProviderResponseError
from citely.models import Answer, RetrievalResult, ScoredChunk
from citely.providers.base import Message
from citely.rag.guardrails import validate_query, verify_citations
from citely.rag.prompts import SYSTEM_PROMPT, build_user_message, select_within_budget

if TYPE_CHECKING:
    from citely.providers.base import LLMProvider
    from citely.rag.retriever import Retriever

#: Models wrap JSON in Markdown fences often enough that stripping them is
#: cheaper than a retry.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_FIRST_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

NO_SOURCES = "The corpus does not contain anything relevant to this question."
UNGROUNDED = "The generated answer could not be traced back to the sources."


class _ModelCitation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: int
    quote: str = ""


class _ModelAnswer(BaseModel):
    """The JSON contract we ask the model to fill in."""

    model_config = ConfigDict(extra="ignore")

    answer: str = ""
    citations: list[_ModelCitation] = Field(default_factory=list)
    insufficient_context: bool = False
    reason: str | None = None


def parse_model_answer(text: str) -> _ModelAnswer:
    """Extract the JSON contract from a raw completion.

    Raises:
        ProviderResponseError: if no valid object can be recovered.
    """
    candidate = _FENCE.sub("", text).strip()
    if not candidate:
        raise ProviderResponseError("the model returned an empty response")

    try:
        return _ModelAnswer.model_validate_json(candidate)
    except ValidationError:
        pass

    # Second chance: the model prefixed prose before the object.
    match = _FIRST_OBJECT.search(candidate)
    if match:
        try:
            return _ModelAnswer.model_validate(json.loads(match.group()))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(f"the model returned malformed JSON: {exc}") from exc

    raise ProviderResponseError("the model did not return a JSON object")


class Answerer:
    """Turns a question into a grounded, cited answer or an explicit refusal."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        *,
        max_context_tokens: int = 6000,
        max_answer_tokens: int = 1024,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_context_tokens = max_context_tokens
        self._max_answer_tokens = max_answer_tokens

    async def answer(self, question: str, *, k: int | None = None) -> Answer:
        """Answer a question, or refuse with a reason.

        Raises:
            InvalidQueryError: if the question itself is unusable.
            RetrievalError, ProviderError: on infrastructure failures.
        """
        query = validate_query(question)

        result = await self._retriever.retrieve(query, k=k)
        if result.is_empty:
            return self._refuse(query, NO_SOURCES)

        sources = select_within_budget(result.chunks, max_tokens=self._max_context_tokens)
        completion = await self._llm.complete(
            system=SYSTEM_PROMPT,
            messages=[Message(role="user", content=build_user_message(query, sources))],
            max_tokens=self._max_answer_tokens,
        )
        parsed = parse_model_answer(completion.text)

        if parsed.insufficient_context:
            return self._refuse(query, parsed.reason or NO_SOURCES, model=completion.model)

        verified, _rejected = verify_citations(
            [(c.source, c.quote) for c in parsed.citations],
            sources,
        )

        # A claim the model could not back with a real quote is exactly the
        # hallucination case, so an unverifiable answer becomes a refusal.
        if not verified or not parsed.answer.strip():
            return self._refuse(query, UNGROUNDED, model=completion.model)

        return Answer(
            query=query,
            text=parsed.answer.strip(),
            citations=verified,
            model=completion.model,
        )

    @staticmethod
    def _refuse(query: str, reason: str, model: str | None = None) -> Answer:
        return Answer(
            query=query,
            text=reason,
            refused=True,
            refusal_reason=reason,
            model=model,
        )


def retrieved_sources(result: RetrievalResult, *, max_tokens: int) -> list[ScoredChunk]:
    """Expose the budgeting step for callers that want to show what was used."""
    return select_within_budget(result.chunks, max_tokens=max_tokens)
