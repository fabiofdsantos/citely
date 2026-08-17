# citely

**Retrieval-Augmented Generation (RAG) as a Service** — provider-agnostic, with
every claim traceable to a source.

citely retrieves passages, asks a model to answer using only those passages, then
**verifies every citation against the source text before returning anything**. A
claim the model can't back with a real quote is never shown — the service refuses
instead.

```console
$ citely query "What is the maximum fine for breaching the prohibitions?"
up to 35 000 000 EUR or, if the offender is an undertaking, up to 7 % of its
total worldwide annual turnover for the preceding financial year [1].

Sources:
  [1] Article 99: Penalties — "Non-compliance with the prohibition of the AI
      practices referred to in Article 5 shall be subject to administrative
      fines of up to 35 000 000 EUR..."
      data/corpus/article-99-penalties.md#b08adc59688340d0

$ citely query "What is the capital of France?"
The corpus does not mention France, so it cannot answer a question about that.
```

[![CI](https://github.com/fabiofdsantos/citely/actions/workflows/ci.yml/badge.svg)](https://github.com/fabiofdsantos/citely/actions/workflows/ci.yml)

## Why

Most RAG demos retrieve chunks, paste them into a prompt, and print whatever
comes back. The model usually stays grounded — and when it doesn't, nothing
notices. The failure is silent, fluent, and confident.

citely closes that gap with one mechanism: **the model must supply a verbatim
quote for every claim, and each quote is checked against the chunk it cites.**

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

## Quickstart

```bash
git clone https://github.com/fabiofdsantos/citely && cd citely
cp .env.example .env      # fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
docker compose up
```

The repo ships a curated extract of the **EU AI Act** as the default corpus, so
there's nothing to download:

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{}'
curl -X POST localhost:8000/query  -H 'content-type: application/json' \
     -d '{"question":"What does Article 5 prohibit?"}'
```

API docs at <http://localhost:8000/docs>. Blank keys in `.env` make the service
refuse to start and name the missing one.

**Without Docker:**

```bash
make install && make check
citely ingest && citely query "what does article 5 prohibit?"
```

**Without API keys** — [Ollama](https://ollama.com), documents never leave your
machine:

```bash
docker compose --profile local up -d ollama
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull nomic-embed-text
```

Then set `CITELY_LLM_PROVIDER=ollama`, `CITELY_EMBEDDING_PROVIDER=ollama` and
`CITELY_LLM_MODEL=llama3.2:3b` in `.env`. Note: Docker on macOS has no GPU
access, so inference is CPU-only and slow; `brew install ollama` gets Metal.

**Your own documents:**

```bash
CITELY_CORPUS_PATH=./my-docs citely ingest
```

Any directory of `.txt`, `.md`, `.markdown` or `.rst`. Re-running is cheap:
unchanged content is never re-embedded, deleted content is dropped from the
index.

## Architecture

```
   citely query ──┐                          ┌──────────────┐
                  ├──▶  Answerer  ──────────▶│  Embeddings  │
   POST /query  ──┘         │                │   provider   │
                            ▼                └──────────────┘
              ┌──────────────────────────┐          │
              │ 1. validate the question │          ▼
              │ 2. retrieve top-k chunks │   ┌──────────────┐
              │ 3. check scope           │◀──│ Vector store │
              │ 4. fit a token budget    │   │ Chroma       │
              │ 5. generate (JSON)       │   │ pgvector     │
              │ 6. VERIFY every quote    │   └──────────────┘
              │ 7. answer, or refuse     │          ▲
              └──────────────────────────┘          │
                            │              ┌──────────────┐
                            ▼              │   Ingestion  │
                   ┌─────────────────┐     │ load → chunk │
                   │  LLM provider   │     │ → embed →    │
                   │ Anthropic /     │     │ upsert       │
                   │ OpenAI / Ollama │     └──────────────┘
                   └─────────────────┘      citely ingest
```

Every boundary is a `Protocol` — `LLMProvider`, `EmbeddingProvider`,
`VectorStore`, `Retriever`. Backends are chosen by config in one registry per
layer; nothing else imports a concrete provider.

```
src/citely/
├── config.py    Settings, validated at startup
├── models.py    Chunk, Citation, Answer
├── errors.py    Error hierarchy → HTTP statuses
├── providers/   LLM + embedding backends
├── stores/      Chroma (default), pgvector
├── ingest/      Loading, chunking, incremental pipeline
├── rag/         Retrieval, prompts, guardrails, verification
├── api/         FastAPI app and wire schemas
└── cli.py       citely ingest | query | status
```

## Design decisions

**Citations carry quotes, not just references.** A `[1]` marker proves a marker
exists; a quote can be checked. Whitespace and Unicode are normalised before
comparison — rewrapping isn't fabrication — but paraphrases are rejected, even
semantically correct ones, because semantic closeness is what hallucination
looks like.

**Refusal is a success, and a 200.** "The corpus can't answer this" is an
expected result clients must render, not an error.

**Chunk ids are content hashes, excluding position.**
`sha256(document_id, text)`, so editing paragraph 2 leaves paragraphs 3–500
untouched and re-ingestion re-embeds only what changed. A re-run over an
unchanged corpus costs zero embedding calls — asserted against the embedder's
call counter, not a log line.

**Mixing embedding models is fatal.** A collection records the model and
dimensionality of its first write and hard-fails on divergence, rather than
silently returning neighbours from an incompatible vector space.

**Retrieved content is untrusted.** Sources are fenced in `<source>` tags,
labelled as data the model must never obey, and the question goes *after* them.
Defence in depth, not proof — the real backstop is that a hijacked answer still
can't produce verifiable quotes.

**A scope check runs before generation.** Verification proves grounding, not
relevance: asked about the *UK* AI Act, an early version answered with EU
requirements, correctly cited. Nothing downstream can catch that, so
[`rag/scope.py`](src/citely/rag/scope.py) refuses when the question names
something no retrieved source mentions. Deliberately conservative — five of its
twenty tests assert it does *not* fire on answerable questions.

**Token budget, not fixed `k`.** A fixed count overflows small context windows
and wastes large ones. The top hit is always kept.

**Structure-aware chunking.** Paragraphs, then sentences, then hard slices only
for text with no internal boundary.

## Configuration

Environment variables, validated once at startup. Full list in
[.env.example](.env.example).

| Variable | Default | Notes |
|---|---|---|
| `CITELY_LLM_PROVIDER` | `anthropic` | `anthropic` · `openai` · `ollama` |
| `CITELY_EMBEDDING_PROVIDER` | `openai` | `openai` · `ollama` (Anthropic has no embeddings API) |
| `CITELY_VECTOR_STORE` | `chroma` | `chroma` · `pgvector` |
| `CITELY_CORPUS_PATH` | `./data/corpus` | Your documents |
| `CITELY_TOP_K` | `6` | Chunks retrieved per query |
| `CITELY_MIN_SCORE` | `0.0` | Drop matches below this similarity |
| `CITELY_SCOPE_CHECK` | `true` | Refuse when the question names something no source mentions |
| `CITELY_MAX_CONTEXT_TOKENS` | `6000` | Lower it for small local models |
| `CITELY_LOG_FORMAT` | `json` | `json` · `console` |

Keys are also read from plain `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, held as
`SecretStr`, and never appear in logs or tracebacks.

**Postgres instead of Chroma:**

```bash
docker compose --profile postgres up -d postgres
export CITELY_VECTOR_STORE=pgvector
export CITELY_PGVECTOR_DSN=postgresql://citely:citely@localhost:5432/citely
```

Schema is created on first use, with an HNSW index over cosine distance.

## Evaluation

```bash
make eval        # offline stubs — no API key, runs in CI
make eval-live   # the configured providers, gated on every metric
```

Twenty cases — ten answerable, ten that must be refused — declaring expected
*behaviour*, not exact strings. They include multi-hop questions, near-miss
distractors, half-answerable questions, and a document carrying a live
prompt-injection payload.

Measured with `llama3.2:3b` + `nomic-embed-text`:

| Metric | Score | |
|---|---|---|
| `retrieval_hit_rate` | 1.00 | Did the right source come back? |
| `answer_accuracy` | 0.80 | Answered when it should |
| `refusal_accuracy` | 1.00 | Refused when it should |
| `citation_precision` | 0.91 | Claimed citations that survived verification |
| `groundedness` | **1.00** | Every answer cites ≥1 verified source (gated) |
| `injection_resistance` | **1.00** | No planted canary ever echoed (gated) |

**The 0.91 is the number that matters.** A real model fabricated a quote,
verification rejected it, and the claim never reached the caller — the premise
of this project, measured rather than asserted.

Both remaining failures are over-refusals (the safe direction): a two-hop
question the model wouldn't complete, and a half-answerable question it refused
entirely instead of answering the half it could.

An eval suite that can't fail is theatre, so the harness is itself tested
against a deliberately fabricating model and asserted to go **red**
([tests/test_evals.py](tests/test_evals.py)). CI gates only the guardrail
metrics offline; model quality is `make eval-live`.

## Development

```bash
make help        # all targets
make check       # lint + type-check + test + eval, same as CI
```

Python 3.11+, mypy strict, ruff, 215 tests at 96% coverage. Backends are tested
twice: with fakes for logic, and against real Chroma and real Postgres for
behaviour. CI runs both, plus pre-commit and a coverage gate.

```bash
# pgvector tests need a database; they skip cleanly without one
docker run -d --name citely-pg -p 5433:5432 \
  -e POSTGRES_USER=citely -e POSTGRES_PASSWORD=citely -e POSTGRES_DB=citely \
  pgvector/pgvector:pg16
CITELY_TEST_PGVECTOR_DSN=postgresql://citely:citely@localhost:5433/citely \
  uv run pytest -m pgvector
```

## Limitations

**Measured on one small local model.** Claude and GPT are unmeasured — they'd
likely score higher, but "likely" isn't a measurement. The corpus is also a
ten-article extract of clean legal prose, not the full Regulation.

**Grounded ≠ relevant.** Verification proves a quote came from the corpus, never
that it answers the question. The scope check covers questions that *name*
something absent; it can't help when a question is off-target without naming
anything.

**Dense retrieval only.** No hybrid search, BM25, or reranking. Questions phrased
far from the source text miss — producing an honest refusal, but a miss.

**Also:** no JSON repair retry on malformed model output; verified citations can
be a subset of the `[n]` markers in the prose; token counting is a `len // 4`
heuristic; no auth or rate limiting, so `/ingest` shouldn't be public as-is;
text files only (no PDF); the image is ~570MB, mostly chromadb's unused ONNX
runtime.

### Next, in order

1. Run the same suite against Claude and GPT — the interesting question is
   whether the *failure modes* match, not which model wins.
2. Hybrid retrieval with a reranker: the likely fix for both over-refusals.
3. Answer the answerable half of a partly-covered question. Needs a third state
   in the `Answer` model, not a prompt tweak.
4. JSON repair retry, plus provider-native structured output.
5. Auth and per-key rate limiting before exposing this to anyone.
6. Grow the golden set to ~50 cases and swap in the full legal text.

## License

MIT. See [LICENSE](LICENSE).
