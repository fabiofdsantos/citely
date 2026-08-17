# Design notes

Why citely is built the way it is. The [README](../README.md) covers what it
does; this covers the decisions behind it and the trade-offs each one makes.

## Verification, not trust

The model must supply a verbatim quote for every claim, and each quote is
checked against the chunk it cites before the answer is returned.

- Quote appears in the cited source → citation kept
- Paraphrased, reworded, or invented → citation dropped
- No citations survive → explicit refusal

Enforced in code, not requested in a prompt. The `Answer` type cannot represent
an uncited answer:

```python
# models.py — a validator, not a convention
if self.refused:
    if not self.refusal_reason:
        raise ValueError("a refused answer must carry a refusal_reason")
elif not self.citations:
    raise ValueError("a non-refused answer must cite at least one source")
```

**Citations carry quotes, not just references.** A `[1]` marker proves a marker
exists; a quote can be checked. Whitespace and Unicode are normalised before
comparison — rewrapping isn't fabrication — but paraphrases are rejected, even
semantically correct ones, because semantic closeness is exactly what
hallucination looks like.

**Refusal is a success, and a 200.** "The corpus can't answer this" is an
expected result clients must render, not an error.

## Retrieval and scope

**A scope check runs before generation.** Verification proves grounding, not
relevance: asked about the *UK* AI Act, an early version answered with EU
requirements, correctly cited, every claim genuinely in the corpus. Nothing
downstream can catch that, so [`rag/scope.py`](../src/citely/rag/scope.py)
refuses when the question names something no retrieved source mentions.

Deliberately conservative — the risk is trading a rare wrong answer for frequent
wrong refusals, so only terms absent from *every* retrieved chunk count, generic
domain words are ignored, and five of its twenty tests assert it does *not* fire
on answerable questions. Measured effect: `refusal_accuracy` 0.90 → 1.00 with
`answer_accuracy` unchanged.

**Token budget, not fixed `k`.** A fixed count overflows a small model's context
window and wastes a large one's. The top hit is always kept, even over budget.

**Structure-aware chunking.** Paragraphs, then sentences, then hard slices only
for text with no internal boundary. Fixed-width slicing produces citations that
quote half-sentences and embeddings of text that means nothing alone.

## Ingestion

**Chunk ids are content hashes, deliberately excluding position.**
`sha256(document_id, text)`, so editing paragraph 2 leaves paragraphs 3–500 with
unchanged ids and re-ingestion re-embeds only what actually changed. A re-run
over an unchanged corpus costs zero embedding calls — asserted against the
embedder's call counter, not a log line.

The trade: two byte-identical chunks in one document collapse to one row. They
are interchangeable as citation targets, so this is acceptable.

**Mixing embedding models is fatal.** A collection records the model and
dimensionality of its first write and hard-fails on divergence. Without it,
switching `CITELY_EMBEDDING_MODEL` returns nearest neighbours from an
incompatible vector space — search still works, and the results are meaningless.
Silent quality collapse is the worst failure class in RAG, so it became a
startup error.

## Prompt injection

**Retrieved content is untrusted input.** Sources are fenced in `<source>` tags,
labelled as data the model must never obey, and the question is placed *after*
them so no document can position itself after the instruction it wants to
override.

Defence in depth, not proof. The real backstop is that a hijacked answer still
cannot produce verifiable quotes. The eval corpus contains a live payload
instructing the model to emit a canary phrase; `injection_resistance` is gated
at 1.0.

## Boundaries

Every boundary is a `Protocol` — `LLMProvider`, `EmbeddingProvider`,
`VectorStore`, `Retriever`. Backends are chosen by config in one registry per
layer; nothing else in the codebase imports a concrete provider.

**Two protocols for providers, not one.** Anthropic ships no embeddings API, so
a single `Provider` interface would force every implementation to lie about half
its surface. Splitting them also lets chat and embedding backends be chosen
independently, which is the common production setup.

**Ollama reuses the OpenAI client.** Its `/v1` endpoint is OpenAI-compatible, so
it is a base URL and a placeholder credential, not a third backend. Free
consequence: vLLM, LM Studio and Azure work through `CITELY_OPENAI_BASE_URL`
with no code change.

**Two stores, one protocol, one test suite shape.** Chroma needs no setup and
makes the quickstart real; pgvector is for teams already running Postgres. Both
normalise cosine distance to a `[0, 1]` score so `CITELY_MIN_SCORE` means the
same thing on either.

**Retries are delegated to the provider SDKs.** Both implement bounded backoff
with jitter and honour `Retry-After`. Hand-rolling would be strictly worse.

**No SDK exception escapes.** Every provider error is mapped onto citely's own
hierarchy, so callers never import `openai` to handle a failure — otherwise the
abstraction is decorative.

## Testing

Backends are tested twice: with fakes for logic, and against real Chroma and
real Postgres for behaviour. The second layer has repeatedly earned itself —
bugs it caught that no fake would have:

- **IVFFlat on an empty table returns nothing.** The index was created before
  any rows existed, so its lists were untrained and searches silently returned
  zero results. Now HNSW, which builds incrementally.
- **`"does not exist"` substring matching swallowed real errors.** Meant to
  catch a missing table on first run, it also caught undefined *columns* and
  *functions*, so a broken query returned an empty list indistinguishable from
  an empty corpus. Now matched by SQLSTATE.
- **Chroma rejects `modify()` payloads containing `hnsw:*`**, because the
  distance function is fixed at creation.

**SQL is composed, never interpolated.** `sql.SQL(...).format(sql.Identifier(...))`
throughout, with values always bound as parameters. Ruff's `S608` can't
distinguish composition from f-string interpolation, so it is suppressed for
that file with the reason recorded in `pyproject.toml`.

**An eval suite that can't fail is theatre.** The harness is itself tested
against a deliberately fabricating model and asserted to go red
([tests/test_evals.py](../tests/test_evals.py)).

## Known limits of the approach

**Grounded is not the same as relevant.** Verification proves a quote came from
the corpus, never that it answers the question. The scope check covers questions
that *name* something absent from the corpus; it cannot help when a question is
off-target without naming anything — "how do I appeal this decision?" against a
corpus describing obligations, say.

**The golden set is small and the corpus is clean.** Twenty cases over ten
articles of well-structured legal prose. Real documents with cross-references,
tables and enumerated subsections are harder, and 1.00 versus 0.90 is one case
apart at this size.
