import json
from pathlib import Path

from app.config import Settings
from app.store import VectorStore


def main():
    corpus = Path(__file__).resolve().parents[1] / "data" / "corpus"
    manifest = json.loads((corpus / "manifest.json").read_text())
    missing = [item["filename"] for item in manifest if not (corpus / item["filename"]).exists()]
    if missing:
        raise SystemExit("Corpus files missing. Run: python -m scripts.fetch_corpus")
    store = VectorStore(Settings())
    for item in manifest:
        result = store.ingest(
            item["filename"], (corpus / item["filename"]).read_bytes(), source=item["source"]
        )
        label = "Already indexed" if result.duplicate else "Indexed"
        print(f"{label}: {result.document.title} ({result.document.chunk_count} chunks)")


if __name__ == "__main__":
    main()
