# citely

Provider-agnostic RAG service that answers questions over a document corpus with
**grounded, cited answers** — every claim traceable to a source chunk, and an
explicit refusal when the corpus can't support an answer.

> Status: under construction. This README grows with the implementation.

## Quickstart

```bash
make install
make check
```

`make help` lists the rest.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
