# Demo corpus

`corpus/` holds a curated extract of **Regulation (EU) 2024/1689** — the EU
Artificial Intelligence Act — as the default corpus so that `citely ingest`
works on a clean checkout with no downloads.

## What is here

Ten articles, chosen to cover the questions a demo actually gets asked, and to
include enough cross-referencing that retrieval has real work to do:

| File | Article |
|---|---|
| `article-01-subject-matter.md` | 1 — Subject matter |
| `article-05-prohibited-practices.md` | 5 — Prohibited AI practices |
| `article-06-high-risk-classification.md` | 6 — Classification rules for high-risk systems |
| `article-09-risk-management.md` | 9 — Risk management system |
| `article-11-technical-documentation.md` | 11 — Technical documentation |
| `article-12-record-keeping.md` | 12 — Record-keeping |
| `article-14-human-oversight.md` | 14 — Human oversight |
| `article-43-conformity-assessment.md` | 43 — Conformity assessment |
| `article-50-transparency.md` | 50 — Transparency obligations |
| `article-99-penalties.md` | 99 — Penalties |

This is an **extract, not the complete Regulation**. Annexes, recitals and most
articles are absent — which is realistic, and useful: questions about missing
provisions are exactly the ones citely should refuse rather than answer.

## Source and licence

Text of Regulation (EU) 2024/1689 of the European Parliament and of the Council
of 13 June 2024 (the AI Act), Official Journal of the European Union.

- Official text: <https://eur-lex.europa.eu/eli/reg/2024/1689/oj>
- Reuse of EUR-Lex content is permitted under Commission Decision 2011/833/EU.

Reproduced for demonstration purposes. Only the authentic version published in
the Official Journal has legal effect; do not rely on this extract for
compliance.

## Using your own documents

Point citely somewhere else and ingest:

```bash
CITELY_CORPUS_PATH=./my-docs citely ingest
```

Nothing here is special — it is a directory of `.md` files.
