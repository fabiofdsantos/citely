"""Deterministic stand-ins for a provider, used by ``make eval --offline``.

These are not a substitute for evaluating a real model. They exist so the
harness itself — retrieval, prompt assembly, JSON parsing, citation
verification, metric computation — runs in CI with no API key and no network,
and so a regression in the *guardrails* fails the build.

Both are genuinely functional rather than canned: the embedder is a real
bag-of-words hashing vectoriser and the "model" is a real extractive
summariser. They are simply bad, which is useful — a weak model exercises the
refusal and verification paths far harder than a strong one.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from typing import Any

from citely.providers.base import Completion, TokenUsage

_DIMENSIONS = 512
_TOKEN = re.compile(r"[a-z0-9]+")
_SOURCE = re.compile(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)
_QUESTION = re.compile(r"<question>\n(.*?)\n</question>", re.DOTALL)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

#: Words carrying no topical signal. Without this the overlap score is dominated
#: by "what", "the", "is" and every source looks equally relevant.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "they",
        "this",
        "to",
        "under",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "must",
        "may",
        "can",
        "should",
        "would",
        "about",
    ]
)

#: Minimum question-to-sentence overlap before the stub will answer at all.
#: Tuned so that questions about absent topics fall through to a refusal.
_ANSWER_THRESHOLD = 0.34


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.casefold()) if t not in _STOPWORDS]


def _bucket(token: str) -> int:
    # hashlib, not hash(): Python salts str hashing per process, which would
    # make embeddings differ between runs and quietly break persistence.
    return int.from_bytes(hashlib.blake2s(token.encode(), digest_size=4).digest()) % _DIMENSIONS


class HashingEmbedder:
    """A bag-of-words hashing vectoriser. Satisfies ``EmbeddingProvider``."""

    def __init__(self, model: str = "offline-hashing-v1") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * _DIMENSIONS
        for token in _tokens(text):
            vector[_bucket(token)] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def aclose(self) -> None:
        return None


class ExtractiveLLM:
    """Answers by quoting the best-matching sentence. Satisfies ``LLMProvider``.

    Because it only ever emits spans copied verbatim from a source, its
    citations always verify — which is the point. Any drop in citation
    precision during an offline run means the verification code changed, not
    that the model got worse.
    """

    def __init__(self, model: str = "offline-extractive-v1") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        prompt = messages[-1].content
        question = _QUESTION.search(prompt)
        sources = _SOURCE.findall(prompt)

        best = self._best_sentence(question.group(1) if question else "", sources)
        if best is None:
            payload = {
                "answer": "",
                "citations": [],
                "insufficient_context": True,
                "reason": "no sentence in the retrieved sources addresses the question",
            }
        else:
            number, sentence = best
            payload = {
                "answer": f"{sentence} [{number}]",
                "citations": [{"source": number, "quote": sentence}],
                "insufficient_context": False,
                "reason": None,
            }

        return Completion(
            text=json.dumps(payload),
            model=self._model,
            usage=TokenUsage(input_tokens=len(prompt) // 4, output_tokens=32),
        )

    @staticmethod
    def _best_sentence(question: str, sources: Sequence[tuple[str, str]]) -> tuple[int, str] | None:
        wanted = set(_tokens(question))
        if not wanted:
            return None

        best: tuple[float, int, str] | None = None
        for number, body in sources:
            for sentence in _SENTENCE.split(body):
                sentence = sentence.strip()
                tokens = set(_tokens(sentence))
                if not tokens:
                    continue
                # Overlap normalised by the question, not the sentence: we want
                # "covers what was asked", not "is short".
                score = len(wanted & tokens) / len(wanted)
                if best is None or score > best[0]:
                    best = (score, int(number), sentence)

        if best is None or best[0] < _ANSWER_THRESHOLD:
            return None
        return best[1], best[2]

    async def aclose(self) -> None:
        return None
