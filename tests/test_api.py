import asyncio
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import ProviderError
from app.llm import JsonLLM
from app.main import create_app
from app.store import VectorStore
from tests.fakes import FakeLLM
from tests.fakes import test_embeddings as embed

DOCUMENT = b"# Request Body\nUse a Pydantic model for the request body."


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path)
    store = VectorStore(settings, embedding_function=embed)
    with TestClient(create_app(settings, store=store, llm=FakeLLM())) as client:
        yield client


def upload(client):
    response = client.post("/ingest", files={"file": ("body.md", DOCUMENT, "text/markdown")})
    assert response.status_code == 201
    return response.json()


def test_ingest_query_feedback_round_trip(client):
    first = upload(client)
    assert upload(client)["duplicate"]
    assert client.get("/documents").json() == [first["document"]]
    response = client.post("/query", json={"question": "How do I define a request body?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["sources"][0]["source"] == "body.md"
    assert body["sources"][0]["citation"] == 1
    feedback = client.post(
        "/feedback",
        json={
            "query_id": body["query_id"],
            "rating": "up",
            "comment": "Helpful",
        },
    )
    assert feedback.status_code == 201
    with sqlite3.connect(client.app.state.repository.path) as db:
        assert db.execute("SELECT count(*) FROM queries").fetchone()[0] == 1
        assert db.execute("SELECT comment FROM feedback").fetchone()[0] == "Helpful"


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001, None])
def test_invalid_question(client, question):
    assert client.post("/query", json={"question": question}).status_code == 422


def test_unknown_feedback_and_invalid_rating(client):
    assert client.post("/feedback", json={"query_id": "unknown", "rating": "up"}).status_code == 404
    invalid = client.post("/feedback", json={"query_id": "unknown", "rating": "bad"})
    assert invalid.status_code == 422


def test_empty_corpus_returns_actionable_error(client):
    response = client.post("/query", json={"question": "Anything?"})
    assert response.status_code == 409
    assert "ingestion" in response.json()["detail"]


def test_invalid_file_and_missing_upload(client):
    assert client.post("/ingest", files={"file": ("x.pdf", b"x")}).status_code == 422
    assert client.post("/ingest").status_code == 422


def test_upload_limit(tmp_path):
    settings = Settings(data_dir=tmp_path, max_upload_bytes=10)
    store = VectorStore(settings, embedding_function=embed)
    with TestClient(create_app(settings, store=store, llm=FakeLLM())) as client:
        assert client.post("/ingest", files={"file": ("x.txt", b"x" * 11)}).status_code == 413


def test_filename_is_metadata_not_a_filesystem_path(client):
    response = client.post("/ingest", files={"file": ("../../outside.md", DOCUMENT)})
    assert response.json()["document"]["source"] == "outside.md"


def test_missing_key_is_reported_without_crashing_startup(tmp_path):
    settings = Settings(data_dir=tmp_path, llm_api_key="")
    store = VectorStore(settings, embedding_function=embed)
    with TestClient(create_app(settings, store=store, llm=JsonLLM(settings))) as client:
        assert client.get("/health").status_code == 200
        upload(client)
        response = client.post("/query", json={"question": "Request body?"})
        assert response.status_code == 503
        assert "LLM_API_KEY" in response.json()["detail"]


def test_provider_error_returns_502(tmp_path):
    class BrokenLLM(FakeLLM):
        async def ask(self, *args):
            raise ProviderError("Provider unavailable")

    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings, store=VectorStore(settings, embed), llm=BrokenLLM())) as c:
        upload(c)
        assert c.post("/query", json={"question": "Request body?"}).status_code == 502


def test_query_deadline_returns_504(tmp_path):
    class SlowLLM(FakeLLM):
        async def ask(self, *args):
            await asyncio.sleep(1)

    settings = Settings(data_dir=tmp_path, query_timeout_seconds=0.05)
    with TestClient(create_app(settings, store=VectorStore(settings, embed), llm=SlowLLM())) as c:
        upload(c)
        assert c.post("/query", json={"question": "Request body?"}).status_code == 504


def test_home_and_openapi(client):
    home = client.get("/")
    assert home.status_code == 200
    assert "Answers with evidence" in home.text
    schema = client.get("/openapi.json").json()
    assert {"/query", "/ingest", "/documents", "/feedback"} <= schema["paths"].keys()
