"""Check real MiniLM retrieval against the included corpus without an LLM API key."""

import json
import tempfile
from pathlib import Path

from app.config import Settings
from app.store import VectorStore


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data/corpus/manifest.json").read_text())
    cases = json.loads((root / "examples/evaluation.json").read_text())
    with tempfile.TemporaryDirectory(prefix="docs-retrieval-") as directory:
        settings = Settings(data_dir=Path(directory), embedding_backend="local")
        store = VectorStore(settings)
        for item in manifest:
            store.ingest(
                item["filename"],
                (root / "data/corpus" / item["filename"]).read_bytes(),
                source=item["source"],
            )
        print(f"Indexed {len(store.list_documents())} documents, {store.collection.count()} chunks")
        for case in cases:
            if not case["expected_source"]:
                continue  # Irrelevant questions require LLM grading, not vector distance alone.
            hits = store.search(case["question"], settings.top_k)
            matched = any(case["expected_source"] in hit.source for hit in hits)
            print(f"{'PASS' if matched else 'FAIL'} | {case['question']}")
            if not matched:
                raise SystemExit("Expected source missing from top-k results")


if __name__ == "__main__":
    main()
