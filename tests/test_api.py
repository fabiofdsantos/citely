"""API tests, exercised through the real ASGI app.

Every test runs the app's full lifespan — startup wiring, middleware, exception
handlers — against real Chroma with stub providers. That is the only way to
catch the failures HTTP layers actually have: bad status mapping, dependencies
never built, error bodies that leak internals.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from citely.api.app import create_app
from citely.api.deps import AppState
from citely.config import Settings
from citely.errors import ProviderError, ProviderRateLimitError, StoreError
from citely.rag.answerer import Answerer
from citely.rag.retriever import VectorRetriever
from citely.stores.registry import build_vector_store
from tests.fakes import FakeEmbedder, FakeLLM

ANSWER_JSON = json.dumps(
    {
        "answer": "Article 0 says something [1].",
        "citations": [{"source": 1, "quote": "Article 0 says something."}],
        "insufficient_context": False,
        "reason": None,
    }
)
REFUSAL_JSON = json.dumps(
    {
        "answer": "",
        "citations": [],
        "insufficient_context": True,
        "reason": "the corpus does not cover this",
    }
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "act.md").write_text(
        "# EU AI Act\n\nArticle 0 says something. Article 1 says something else.",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def settings(tmp_path: Path, corpus: Path) -> Settings:
    return Settings(
        llm_provider="ollama",
        embedding_provider="ollama",
        chroma_path=tmp_path / "chroma",
        collection_name="api-test",
        corpus_path=corpus,
        chunk_size=2000,
        log_format="console",
    )


def build_client(settings: Settings, llm: Any = None) -> Iterator[TestClient]:
    """An app whose providers are stubs but whose store and wiring are real."""
    embedder = FakeEmbedder()
    store = build_vector_store(settings)
    chat = llm or FakeLLM(ANSWER_JSON)
    state = AppState(
        settings=settings,
        embedder=embedder,
        llm=chat,
        store=store,
        answerer=Answerer(VectorRetriever(embedder, store, top_k=settings.top_k), chat),
    )

    with TestClient(create_app(settings, state=state)) as client:
        yield client


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    yield from build_client(settings)


class TestHealth:
    def test_healthz_reports_the_index(self, client: TestClient) -> None:
        response = client.get("/healthz")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["backend"] == "chroma"
        assert body["chunk_count"] == 0

    def test_healthz_reflects_ingested_content(self, client: TestClient) -> None:
        client.post("/ingest", json={})

        body = client.get("/healthz").json()
        assert body["chunk_count"] > 0
        assert body["embedding_model"] == "fake-embed"


class TestIngest:
    def test_ingest_reports_what_changed(self, client: TestClient) -> None:
        response = client.post("/ingest", json={})

        assert response.status_code == 200
        body = response.json()
        assert body["embedded"] > 0
        assert body["unchanged"] is False
        assert body["documents"][0]["source_uri"].endswith("act.md")

    def test_second_ingest_is_a_no_op(self, client: TestClient) -> None:
        client.post("/ingest", json={})

        body = client.post("/ingest", json={}).json()

        assert body["embedded"] == 0
        assert body["unchanged"] is True

    def test_missing_path_is_a_400_with_a_code(self, client: TestClient) -> None:
        response = client.post("/ingest", json={"path": "/nonexistent/corpus"})

        assert response.status_code == 400
        assert response.json()["code"] == "ingestion_error"

    def test_unknown_fields_are_rejected(self, client: TestClient) -> None:
        assert client.post("/ingest", json={"pathh": "typo"}).status_code == 422


class TestQuery:
    def test_grounded_answer_carries_numbered_citations(self, client: TestClient) -> None:
        client.post("/ingest", json={})

        response = client.post("/query", json={"question": "what does article 0 say?"})

        assert response.status_code == 200
        body = response.json()
        assert body["refused"] is False
        assert body["citations"][0]["number"] == 1
        assert body["citations"][0]["quote"] == "Article 0 says something."
        assert body["citations"][0]["source_uri"].endswith("act.md")

    def test_refusal_is_a_200_not_an_error(self, settings: Settings) -> None:
        """A refusal is a valid result clients must render, not a failure."""
        for client in build_client(settings, llm=FakeLLM(REFUSAL_JSON)):
            client.post("/ingest", json={})

            response = client.post("/query", json={"question": "capital of France?"})

            assert response.status_code == 200
            assert response.json()["refused"] is True
            assert response.json()["citations"] == []

    def test_empty_question_is_a_422(self, client: TestClient) -> None:
        assert client.post("/query", json={"question": ""}).status_code == 422

    def test_oversized_question_is_a_422(self, client: TestClient) -> None:
        assert client.post("/query", json={"question": "a" * 5000}).status_code == 422

    def test_whitespace_question_is_a_422_from_the_guardrail(self, client: TestClient) -> None:
        """Passes schema validation, then fails the guardrail — still a 422."""
        response = client.post("/query", json={"question": "   "})

        assert response.status_code == 422
        assert response.json()["code"] == "invalid_query"


class TestErrorMapping:
    """Infrastructure failures must not surface as 500s with stack traces."""

    @pytest.mark.parametrize(
        ("error", "expected_status", "expected_code"),
        [
            (ProviderRateLimitError("slow down"), 429, "provider_rate_limit"),
            (StoreError("chroma is gone"), 503, "store_error"),
        ],
    )
    def test_errors_map_to_status_and_code(
        self,
        client: TestClient,
        error: Exception,
        expected_status: int,
        expected_code: str,
    ) -> None:
        class FailingAnswerer:
            async def answer(self, question: str, *, k: int | None = None) -> Any:
                raise error

        client.app.state.citely.answerer = FailingAnswerer()  # type: ignore[attr-defined]

        response = client.post("/query", json={"question": "what does article 0 say?"})

        assert response.status_code == expected_status
        assert response.json() == {"code": expected_code, "message": str(error)}


class TestContract:
    def test_openapi_documents_both_endpoints(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/query", "/ingest", "/healthz"} <= paths.keys()

    def test_request_id_is_echoed(self, client: TestClient) -> None:
        """Correlating a client report with server logs needs a shared id."""
        response = client.get("/healthz", headers={"x-request-id": "abc123"})
        assert response.headers["x-request-id"] == "abc123"

    def test_request_id_is_generated_when_absent(self, client: TestClient) -> None:
        assert client.get("/healthz").headers["x-request-id"]


async def test_app_state_closes_providers(settings: Settings) -> None:
    """A leaked HTTP client per restart is a slow, invisible resource leak."""
    closed: list[str] = []

    class TrackingEmbedder(FakeEmbedder):
        async def aclose(self) -> None:
            closed.append("embedder")

    state = AppState.build(settings)
    state.embedder = TrackingEmbedder()
    await state.aclose()

    assert "embedder" in closed


def test_provider_failures_are_not_masked_as_retrieval_errors(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider outage is a 502, not a 503.

    The retriever used to wrap every exception as RetrievalError, which turned
    an upstream 502-class failure into "our store is unavailable" — misleading
    for anyone reading status codes to decide whether to retry.
    """

    class DeadEmbedder(FakeEmbedder):
        async def embed_query(self, text: str) -> list[float]:
            raise ProviderError("provider is unreachable")

    store = build_vector_store(settings)
    embedder = DeadEmbedder()
    llm = FakeLLM(ANSWER_JSON)
    state = AppState(
        settings=settings,
        embedder=embedder,
        llm=llm,
        store=store,
        answerer=Answerer(VectorRetriever(embedder, store), llm),
    )

    with TestClient(create_app(settings, state=state)) as client:
        response = client.post("/query", json={"question": "what does article 0 say?"})

    assert response.status_code == 502
    assert response.json()["code"] == "provider_error"
