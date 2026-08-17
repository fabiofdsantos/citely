# citely

**Retrieval-Augmented Generation (RAG) as a Service** — provider-agnostic, and
built so every claim in an answer is traceable to a source.

Ask a question over your documents. citely retrieves the relevant passages, asks
a model to answer using only those passages, and then **verifies every citation
against the source text before returning anything**. A claim the model cannot
back with a real quote is not shown — the service refuses instead.

```console
$ citely query "What does Article 5 prohibit?"
Article 5 prohibits social scoring of natural persons by public authorities [1].

Sources:
  [1] EU AI Act — "Article 5 sets out prohibited practices."
      data/corpus/ai_act.md#a3f9c1e4b7d2

$ citely query "What is the capital of France?"
The corpus does not contain anything relevant to this question.
```

[![CI](https://github.com/fabiofdsantos/citely/actions/workflows/ci.yml/badge.svg)](https://github.com/fabiofdsantos/citely/actions/workflows/ci.yml)

---

## Why this exists

Most RAG demos retrieve some chunks, paste them into a prompt, and print
whatever comes back. The model usually stays grounded — and when it doesn't,
nothing notices. The failure is silent, fluent, and confident.

citely closes that gap with one mechanism: **the model must supply a verbatim
quote for every claim, and each quote is checked against the chunk it cites.**

- Quote appears in the cited source → citation kept.
- Quote is paraphrased, reworded, or invented → citation dropped.
- No citations survive → the answer becomes an explicit refusal.

This is enforced in code, not requested in a prompt. The `Answer` type itself
cannot represent an uncited answer:

```python
# citely/models.py — a validator, not a convention
if self.refused:
    if not self.refusal_reason:
        raise ValueError("a refused answer must carry a refusal_reason")
elif not self.citations:
    raise ValueError("a non-refused answer must cite at least one source")
```

## Quickstart

```bash
git clone https://github.com/fabiofdsantos/citely && cd citely
cp .env.example .env      # add ANTHROPIC_API_KEY and OPENAI_API_KEY
docker compose up
```

Then:

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{}'
curl -X POST localhost:8000/query  -H 'content-type: application/json' \
     -d '{"question":"What does Article 5 prohibit?"}'
```

Interactive API docs at <http://localhost:8000/docs>.

### Without Docker

```bash
make install        # uv sync + pre-commit hooks
make check          # lint, type-check, tests, evals
citely ingest       # index ./data/corpus
citely query "what does article 5 prohibit?"
citely status       # what is currently indexed
```

### Without API keys

[Ollama](https://ollama.com) runs everything locally, and your documents never
leave the machine:

```bash
ollama pull llama3.1:8b && ollama pull nomic-embed-text
export CITELY_LLM_PROVIDER=ollama CITELY_EMBEDDING_PROVIDER=ollama
citely ingest && citely query "what does article 5 prohibit?"
```

Or in Docker: `docker compose --profile local up`.

### Bring your own documents

```bash
CITELY_CORPUS_PATH=./my-docs citely ingest
```

Any directory of `.txt`, `.md`, `.markdown` or `.rst` files. Re-running is cheap
and safe: unchanged content is never re-embedded, and deleted content is removed
from the index.

## Architecture

```
                       ┌──────────────┐        ┌──────────────┐
   citely query  ──┐   │              │        │  Embeddings  │
                   ├──▶│   Answerer   │───────▶│   provider   │
   POST /query   ──┘   │              │        └──────────────┘
                       └───┬──────────┘                │
                           │                           ▼
              ┌────────────┴─────────────┐     ┌──────────────┐
              │ 1. validate the question │     │ Vector store │
              │ 2. retrieve top-k chunks │◀────│ Chroma       │
              │ 3. fit a token budget    │     │ pgvector     │
              │ 4. generate (JSON)       │     └──────────────┘
              │ 5. VERIFY every quote    │            ▲
              │ 6. answer, or refuse     │            │
              └──────────────────────────┘     ┌──────────────┐
                           │                   │   Ingestion  │
                           ▼                   │ load → chunk │
                  ┌─────────────────┐          │ → embed →    │
                  │ LLM provider    │          │ upsert       │
                  │ Anthropic       │          └──────────────┘
                  │ OpenAI / Ollama │                 ▲
                  └─────────────────┘          citely ingest
                                               POST /ingest
```

Every arrow crossing a boundary is a `Protocol` — `LLMProvider`,
`EmbeddingProvider`, `VectorStore`, `Retriever`. Backends are selected by
configuration and constructed in one registry per layer; nothing else in the
codebase imports a concrete provider.

```
src/citely/
├── config.py       Settings (pydantic-settings), validated at startup
├── models.py       Domain types: Chunk, Citation, Answer
├── errors.py       Error hierarchy with stable codes → HTTP statuses
├── providers/      LLM + embedding backends behind two Protocols
├── stores/         Chroma (default) and pgvector, behind one Protocol
├── ingest/         Loading, structure-aware chunking, incremental pipeline
├── rag/            Retrieval, prompts, guardrails, citation verification
├── api/            FastAPI app, routes, wire schemas
└── cli.py          citely ingest | query | status
evals/              Golden dataset, metrics, runner
```

## Design decisions

**Citations carry quotes, not just references.** A `[1]` marker proves a marker
exists. A quote can be checked against the source, so verification is mechanical
rather than a matter of trust. Whitespace and Unicode are normalised before
comparison — re-wrapping isn't fabrication — but a paraphrase is rejected, even a
semantically correct one, because semantic closeness is what hallucination looks
like.

**Refusal is a success, and a 200.** "The corpus can't answer this" is a valid,
expected outcome that clients must render. Modelling it as an error would push
callers toward ignoring it.

**Chunk ids are content hashes, deliberately excluding position.**
`sha256(document_id, text)` means editing paragraph 2 leaves paragraphs 3–500
with unchanged ids, so re-ingestion re-embeds only what actually changed.
Re-running over an untouched corpus costs zero embedding calls — asserted in the
test suite against the embedder's call counter, not a log line.

**Mixing embedding models is fatal, by design.** A collection records the model
and dimensionality of its first write. Switching models later raises rather than
silently returning nearest neighbours from an incompatible vector space — the
kind of failure that looks like "retrieval got worse" for weeks.

**Retrieved content is untrusted input.** Sources are fenced in `<source>` tags,
labelled as data the model must never obey, and the question is placed *after*
them so no document can position itself after the instruction it wants to
override. This is defence in depth, not a proof — the real backstop is that a
hijacked answer still cannot produce verifiable quotes.

**Token budget, not fixed `k`.** A fixed count overflows a small model's context
window and wastes a large one's. The top hit is always kept, even over budget.

**Structure-aware chunking.** Paragraphs, then sentences, then (only for text
with no internal boundary) hard slices. Fixed-width slicing produces citations
that quote half-sentences.

**Two vector stores, one Protocol, one test suite shape.** Chroma needs no setup
and makes the quickstart real; pgvector is for teams already running Postgres.
Both normalise cosine distance to a `[0, 1]` score so `CITELY_MIN_SCORE` means
the same thing on either.

## Configuration

Everything is environment variables, validated once at startup — a missing key
fails immediately with a readable message, not on the first request. See
[.env.example](.env.example) for the full list.

| Variable | Default | Notes |
|---|---|---|
| `CITELY_LLM_PROVIDER` | `anthropic` | `anthropic` · `openai` · `ollama` |
| `CITELY_EMBEDDING_PROVIDER` | `openai` | `openai` · `ollama` (Anthropic has no embeddings API) |
| `CITELY_VECTOR_STORE` | `chroma` | `chroma` · `pgvector` |
| `CITELY_CORPUS_PATH` | `./data/corpus` | Your documents |
| `CITELY_TOP_K` | `6` | Chunks retrieved per query |
| `CITELY_MIN_SCORE` | `0.0` | Drop matches below this similarity |
| `CITELY_MAX_CONTEXT_TOKENS` | `6000` | Retrieval budget; lower it for small local models |
| `CITELY_CHUNK_SIZE` / `_OVERLAP` | `1000` / `150` | Characters |
| `CITELY_LOG_FORMAT` | `json` | `json` · `console` |

Secrets are read from the conventional `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
too, held as `SecretStr`, and never appear in logs or tracebacks.

### Postgres instead of Chroma

```bash
docker compose --profile postgres up -d postgres
export CITELY_VECTOR_STORE=pgvector
export CITELY_PGVECTOR_DSN=postgresql://citely:citely@localhost:5432/citely
citely ingest
```

The schema is created on first use, with an HNSW index over cosine distance.

## Evaluation

```bash
make eval        # offline stubs — no API key, runs in CI
make eval-live   # the configured providers; gates on every metric
```

The [golden set](evals/golden.yaml) declares expected *behaviour* rather than
expected strings — scoring against fixed sentences measures phrasing and breaks
on every model upgrade. Half the cases must be refused, including the adjacent-
but-absent trap ("What does GDPR Article 17 say?"), a detail the corpus mentions
without quantifying, a request for legal advice, and a direct injection attempt.

| Metric | What it answers |
|---|---|
| `retrieval_hit_rate` | Did the right source come back? Caps everything downstream. |
| `answer_accuracy` / `refusal_accuracy` | Reported separately — over- and under-refusal fail for opposite reasons. |
| `citation_precision` | Of claimed citations, how many survived verification? Fabrication, measured. |
| `groundedness` | Every answer carries ≥1 verified citation. Must be 1.0 — below that is a broken guardrail. |

An eval suite that can't fail is theatre, so the harness is itself tested against
a deliberately fabricating model and asserted to go **red**
([tests/test_evals.py](tests/test_evals.py)).

CI gates only the guardrail metrics offline, because the stubs' answer quality
isn't what CI is checking. Model quality is `make eval-live`.

## Development

```bash
make help        # all targets
make check       # lint + type-check + test + eval, same as CI
make test-live   # tests that hit real provider APIs (needs keys)
```

- Python 3.11+, fully type-hinted, **mypy strict**, no `Any` escapes in the
  domain.
- ruff for lint and format; pre-commit runs both plus secret detection.
- ~190 tests. Provider and store backends are tested twice: with fakes for
  logic, and against real Chroma / real Postgres for behaviour.

```bash
# pgvector tests need a database; they skip cleanly without one
docker run -d --name citely-pg -p 5433:5432 \
  -e POSTGRES_USER=citely -e POSTGRES_PASSWORD=citely -e POSTGRES_DB=citely \
  pgvector/pgvector:pg16
CITELY_TEST_PGVECTOR_DSN=postgresql://citely:citely@localhost:5433/citely \
  uv run pytest -m pgvector
```

## Limitations, honestly

**Not yet verified against a live model.** Every test and the offline evals run
against stubs. The JSON output contract, the quote-verification hit rate, and
the refusal behaviour of a real Claude or GPT are unmeasured until someone runs
`make eval-live` with keys. That is the single biggest gap.

**Dense retrieval only.** No hybrid search, no BM25, no reranking. Questions
phrased very differently from the source text will miss, and the answer will be
an honest refusal rather than a wrong answer — but a miss all the same.

**Verified citations can be a subset of the `[n]` markers in the prose.** If one
citation is dropped, the answer text may still reference it. Renumbering during
response formatting is the fix.

**No repair loop on malformed JSON.** A model that returns unparseable output
raises rather than retrying. Small local models will need that loop.

**Token counting is a `len(text) // 4` heuristic**, not a real tokenizer.
Adequate for budgeting, wrong at the margins for non-English text.

**No auth, no rate limiting, no multi-tenancy.** `/ingest` reads server-side
paths and is not something to expose publicly as-is.

**Text files only.** No PDF, HTML, or DOCX loaders yet — the loader interface is
one function, but the parsers aren't written.

**The image is ~570MB**, mostly chromadb's ONNX runtime dependency, which citely
never uses because embeddings come from a provider.

### What I'd do next, in order

1. Run `make eval-live` against Claude and GPT; publish the numbers, then tune
   the prompt against measured citation precision rather than intuition.
2. Hybrid retrieval (BM25 + dense) with a reranker, measured against the same
   golden set — the fix for the honest-refusal-that-should-have-answered case.
3. A JSON repair retry, plus provider-native structured output where available.
4. Auth and per-key rate limiting before this is exposed to anyone.
5. Grow the golden set to ~50 cases, including adversarial retrieved content.
6. PDF and HTML loaders; the EU AI Act ships as a PDF.

## License

MIT. See [LICENSE](LICENSE).
