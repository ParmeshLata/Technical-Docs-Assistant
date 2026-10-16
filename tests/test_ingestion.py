import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.errors import InvalidDocument
from app.ingestion import parse_document, split_document
from app.store import VectorStore
from tests.fakes import test_embeddings as embed


def test_bundled_corpus_matches_pinned_snapshot():
    corpus = Path(__file__).resolve().parents[1] / "data" / "corpus"
    snapshot = json.loads((corpus / "snapshot.json").read_text())
    assert len(snapshot["documents"]) == 5
    for item in snapshot["documents"]:
        data = (corpus / item["filename"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"], item["filename"]
        assert b"{* ../../docs_src/" not in data


def test_local_embeddings_disable_telemetry_before_initialization(tmp_path, monkeypatch):
    import sys

    import chromadb.utils.embedding_functions as functions

    events = []
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            disable_telemetry_events=lambda: events.append("disabled"),
        ),
    )

    def factory():
        events.append("initialized")
        return lambda texts: [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(functions, "ONNXMiniLM_L6_V2", factory)
    store = VectorStore(Settings(data_dir=tmp_path))
    assert store._embed(["example"]) == [[1.0, 0.0]]
    assert events == ["disabled", "initialized"]


def test_html_strips_navigation_and_retains_code():
    title, text = parse_document(
        "docs.html",
        b"""
        <html><title>Guide</title><nav>irrelevant nav</nav><main>
        <h1>Models</h1><p>Use a model.</p><pre>class Item(BaseModel):\n    name: str</pre>
        <script>bad()</script></main></html>
    """,
    )
    assert title == "Guide"
    assert "# Models" in text and "```" in text and "name: str" in text
    assert "irrelevant nav" not in text and "bad()" not in text


@pytest.mark.parametrize(
    "filename,data",
    [
        ("x.pdf", b"PDF"),
        ("x.md", b"\xff"),
        ("x.txt", b"  "),
        ("x.txt", b"\x00abc"),
    ],
)
def test_invalid_documents(filename, data):
    with pytest.raises(InvalidDocument):
        parse_document(filename, data)


def test_sections_code_and_hard_chunk_limit():
    text = "# First\n```python\n# a code comment\nprint(1)\n```\n\n## Second\n" + "word " * 800
    _, chunks = split_document(text, "Title", "guide.md", 500, 75)
    assert all(len(c.text) <= 500 for c in chunks)
    assert chunks[0].section == "First"
    assert "# a code comment" in chunks[0].text
    assert all(c.section == "Second" for c in chunks[1:])
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_persisted_index_duplicate_and_source_metadata(tmp_path):
    settings = Settings(data_dir=tmp_path)
    store = VectorStore(settings, embedding_function=embed)
    content = b"# Request Body\nUse a Pydantic model for request body validation."
    first = store.ingest("body.md", content, source="https://example.com/body")
    assert not first.duplicate
    assert store.ingest("body.md", content, source="https://example.com/body").duplicate
    reopened = VectorStore(settings, embedding_function=embed)
    assert reopened.list_documents() == [first.document]
    hits = reopened.search("Pydantic body", 10)
    assert len(hits) == 1
    assert hits[0].source == "https://example.com/body"
    assert hits[0].document_id == first.document.document_id


def test_changed_content_is_a_separate_snapshot(tmp_path):
    store = VectorStore(Settings(data_dir=tmp_path), embedding_function=embed)
    store.ingest("body.md", b"Original content")
    store.ingest("body.md", b"New content")
    assert len(store.list_documents()) == 2


def test_embedding_or_chunk_configuration_gets_separate_collection(tmp_path):
    first = VectorStore(Settings(data_dir=tmp_path), embedding_function=embed)
    first.ingest("x.md", b"A small document")
    second = VectorStore(Settings(data_dir=tmp_path, chunk_size=700), embedding_function=embed)
    assert second.list_documents() == []
