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

Most RAG demos print whatever the model returns; when it drifts off-source,
nothing notices. citely makes that failure impossible to ship silently — the
model must quote its sources, and every quote is checked. Unverifiable claims
become refusals. See [docs/design.md](docs/design.md) for the how and why.

## Quickstart

```bash
git clone https://github.com/fabiofdsantos/citely && cd citely
cp .env.example .env      # fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
docker compose up
```

A curated extract of the **EU AI Act** ships as the default corpus, so there's
nothing to download:

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{}'
curl -X POST localhost:8000/query  -H 'content-type: application/json' \
     -d '{"question":"What does Article 5 prohibit?"}'
```

API docs at <http://localhost:8000/docs>.

**Without Docker:** `make install && make check`, then `citely ingest` and
`citely query "..."`.

**Without API keys** — [Ollama](https://ollama.com), documents never leave your
machine:

```bash
docker compose --profile local up -d ollama
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull nomic-embed-text
```

Then set `CITELY_LLM_PROVIDER=ollama`, `CITELY_EMBEDDING_PROVIDER=ollama` and
`CITELY_LLM_MODEL=llama3.2:3b` in `.env`. Docker on macOS has no GPU access, so
this is CPU-only and slow; `brew install ollama` gets Metal.

**Your own documents:** `CITELY_CORPUS_PATH=./my-docs citely ingest`. Any
directory of `.txt`, `.md`, `.markdown` or `.rst`. Re-running is cheap —
unchanged content is never re-embedded, deleted content is dropped.

## How it works

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

Every boundary is a `Protocol`; backends are chosen by config in one registry
per layer. `src/citely/` splits into `providers/`, `stores/`, `ingest/`, `rag/`,
`api/` and `cli.py`.

## Configuration

Environment variables, validated at startup. Full list in
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

Keys are also read from plain `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, held as
`SecretStr`, never logged. For Postgres: `docker compose --profile postgres up
-d postgres`, then set `CITELY_VECTOR_STORE=pgvector` and
`CITELY_PGVECTOR_DSN`.

## Evaluation

```bash
make eval        # offline stubs — no API key, runs in CI
make eval-live   # the configured providers, gated on every metric
```

Twenty cases — ten answerable, ten that must be refused — declaring expected
*behaviour*, not exact strings: multi-hop questions, near-miss distractors,
half-answerable questions, and a document carrying a live prompt-injection
payload. Measured with `llama3.2:3b` + `nomic-embed-text`:

| Metric | Score | |
|---|---|---|
| `retrieval_hit_rate` | 1.00 | Did the right source come back? |
| `answer_accuracy` | 0.80 | Answered when it should |
| `refusal_accuracy` | 1.00 | Refused when it should |
| `citation_precision` | 0.91 | Claimed citations that survived verification |
| `groundedness` | **1.00** | Every answer cites ≥1 verified source (gated) |
| `injection_resistance` | **1.00** | No planted canary ever echoed (gated) |

**The 0.91 is the number that matters.** A real model fabricated a quote,
verification rejected it, and the claim never reached the caller. Both remaining
failures are over-refusals — the safe direction.

## Development

```bash
make help        # all targets
make check       # lint + type-check + test + eval, same as CI
```

Python 3.11+, mypy strict, ruff, 215 tests at 96% coverage. Backends are tested
with fakes for logic and against real Chroma and real Postgres for behaviour;
CI runs both, plus pre-commit and a coverage gate. pgvector tests skip cleanly
without a database — see [docs/design.md](docs/design.md#testing).

## Limitations

**Measured on one small local model.** Claude and GPT are unmeasured. The corpus
is a ten-article extract of clean legal prose, not the full Regulation.

**Grounded ≠ relevant.** Verification proves a quote came from the corpus, never
that it answers the question. The scope check covers questions that *name*
something absent; it can't help when a question is off-target without naming
anything.

**Dense retrieval only.** No hybrid search, BM25, or reranking. Questions phrased
far from the source text miss — an honest refusal, but a miss.

**Also:** no JSON repair retry; verified citations can be a subset of the `[n]`
markers in the prose; token counting is a `len // 4` heuristic; no auth or rate
limiting, so `/ingest` shouldn't be public as-is; text files only; the image is
~570MB, mostly chromadb's unused ONNX runtime.

**Next:** run the suite against Claude and GPT; hybrid retrieval with a
reranker; answer the answerable half of a partly-covered question; JSON repair
retry; auth; grow the golden set to ~50 cases over the full legal text.

## License

MIT. See [LICENSE](LICENSE).
