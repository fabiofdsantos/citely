# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Builder: resolve and install dependencies into a self-contained virtualenv.
# Separating this from the runtime stage keeps uv, build tools and the lock file
# out of the shipped image.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer: application code changes on every
# commit, the dependency set almost never does, so this cache survives.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Runtime: no uv, no build tools, no source of truth for dependencies.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# curl is here only so the container can health-check itself; it is small and
# the alternative is an ambiguous "starting" state in compose.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

# A non-root user: a container that never needs to write outside its data
# volume has no reason to run as root.
RUN useradd --create-home --uid 1000 citely

WORKDIR /app

COPY --from=builder --chown=citely:citely /app/.venv /app/.venv
COPY --chown=citely:citely src/ ./src/
COPY --chown=citely:citely pyproject.toml README.md ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CITELY_CHROMA_PATH=/data/chroma \
    CITELY_CORPUS_PATH=/data/corpus

RUN mkdir -p /data/chroma /data/corpus && chown -R citely:citely /data

USER citely
EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "citely.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
