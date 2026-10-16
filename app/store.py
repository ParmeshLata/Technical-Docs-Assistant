"""Persistent Chroma evidence plus SQLite query/feedback records."""

import hashlib
import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from itertools import zip_longest
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings
from app.errors import ConfigurationError, ProviderError
from app.ingestion import Chunk, parse_document, split_document
from app.schemas import DocumentInfo, IngestResponse, QueryResponse


class Repository:
    def __init__(self, settings: Settings):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = settings.data_dir / "records.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS queries (
                    query_id TEXT PRIMARY KEY, question TEXT NOT NULL,
                    response_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL REFERENCES queries(query_id),
                    rating TEXT NOT NULL CHECK(rating IN ('up','down')),
                    comment TEXT, created_at TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def save_query(self, question: str, response: QueryResponse):
        with self.connect() as db:
            db.execute(
                "INSERT INTO queries VALUES (?, ?, ?, ?)",
                (
                    response.query_id,
                    question,
                    response.model_dump_json(),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def add_feedback(self, query_id: str, rating: str, comment: str | None) -> str | None:
        feedback_id = str(uuid4())
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM queries WHERE query_id=?", (query_id,)).fetchone():
                return None
            db.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?)",
                (
                    feedback_id,
                    query_id,
                    rating,
                    comment,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return feedback_id


class VectorStore:
    def __init__(self, settings: Settings, embedding_function=None):
        self.settings = settings
        self.lock = threading.RLock()
        self.client = chromadb.PersistentClient(
            path=str(settings.data_dir / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Explicit embeddings avoid implicit Chroma downloads during startup/listing.
        self._embedding_function = embedding_function
        embedding_identity = (
            "onnx-all-MiniLM-L6-v2"
            if settings.embedding_backend == "local"
            else f"{settings.embedding_base_url}|{settings.embedding_model}"
        )
        config_id = hashlib.sha256(
            json.dumps(
                {
                    "embedding": embedding_identity,
                    "size": settings.chunk_size,
                    "overlap": settings.chunk_overlap,
                    "splitter": "sections-v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]
        self.collection = self.client.get_or_create_collection(
            name=f"technical_docs_{config_id}",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_function is None:
            if self.settings.embedding_backend == "local":
                import onnxruntime
                from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

                # Disable runtime telemetry before constructing any inference session.
                onnxruntime.disable_telemetry_events()
                self._embedding_function = ONNXMiniLM_L6_V2()
            else:
                from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

                key = self.settings.embedding_api_key.get_secret_value()
                if not key:
                    raise ConfigurationError("Set EMBEDDING_API_KEY for OpenAI embeddings")
                self._embedding_function = OpenAIEmbeddingFunction(
                    api_key=key,
                    model_name=self.settings.embedding_model,
                    api_base=self.settings.embedding_base_url,
                )
        try:
            return [list(map(float, vector)) for vector in self._embedding_function(texts)]
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ProviderError(
                "Embedding failed. Check connectivity, model download, and embedding configuration."
            ) from exc

    def ingest(self, filename: str, data: bytes, source: str | None = None) -> IngestResponse:
        title, text = parse_document(filename, data)
        source = source or filename
        document_id, chunks = split_document(
            text,
            title,
            source,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
        # A single-process lock keeps duplicate writes and reads consistent for the local app.
        with self.lock:
            existing = self.collection.get(where={"document_id": document_id}, limit=1)
            if existing["ids"]:
                metadata = existing["metadatas"][0]
                return IngestResponse(document=self._document(metadata), duplicate=True)
            vectors = []
            for offset in range(0, len(chunks), 64):
                vectors.extend(self._embed([c.text for c in chunks[offset : offset + 64]]))
            now = datetime.now(UTC).isoformat()
            metadata = [
                {
                    "document_id": c.document_id,
                    "title": c.title,
                    "source": c.source,
                    "section": c.section,
                    "position": c.position,
                    "chunk_count": len(chunks),
                    "indexed_at": now,
                }
                for c in chunks
            ]
            self.collection.upsert(
                ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=metadata,
                embeddings=vectors,
            )
            return IngestResponse(document=self._document(metadata[0]), duplicate=False)

    @staticmethod
    def _document(metadata: dict) -> DocumentInfo:
        return DocumentInfo(**{key: metadata[key] for key in DocumentInfo.model_fields})

    def list_documents(self) -> list[DocumentInfo]:
        with self.lock:
            rows = self.collection.get(include=["metadatas"])
        documents = {}
        for metadata in rows["metadatas"] or []:
            documents[metadata["document_id"]] = self._document(metadata)
        return sorted(documents.values(), key=lambda d: (d.title, d.document_id))

    def search(self, query: str, top_k: int) -> list[Chunk]:
        with self.lock:
            count = self.collection.count()
            if count == 0:
                return []
            vector = self._embed([query])[0]
            rows = self.collection.query(
                query_embeddings=[vector],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
        return [
            Chunk(
                chunk_id=chunk_id,
                text=text,
                distance=distance,
                **{
                    key: metadata[key]
                    for key in (
                        "document_id",
                        "title",
                        "source",
                        "section",
                        "position",
                    )
                },
            )
            for chunk_id, text, metadata, distance in zip(
                rows["ids"][0],
                rows["documents"][0],
                rows["metadatas"][0],
                rows["distances"][0],
                strict=True,
            )
        ]

    def search_many(self, queries: list[str], top_k: int) -> list[Chunk]:
        """Interleave original/rephrased search results without increasing LLM grading count.

        Title-only chunks carry no answer evidence. Remove them before merging, then deduplicate
        by chunk ID. Original-question results come first at each rank so named terms remain
        represented even when a rewritten query drifts. Each query is still a vector search.
        """
        unique_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
        candidates = []
        for query in unique_queries:
            hits = self.search(query, min(top_k + 2, 10))
            candidates.append(
                [
                    chunk
                    for chunk in hits
                    if re.sub(r"^#{1,6}\s*", "", chunk.text.strip()).casefold()
                    not in {chunk.title.strip().casefold(), chunk.section.strip().casefold()}
                ]
            )
        selected: list[Chunk] = []
        seen: set[str] = set()
        for row in zip_longest(*candidates):
            for chunk in row:
                if chunk is not None and chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    selected.append(chunk)
                    if len(selected) == top_k:
                        return selected
        return selected
