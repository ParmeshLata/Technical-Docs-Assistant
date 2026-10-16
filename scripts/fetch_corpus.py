"""Fetch a pinned FastAPI documentation snapshot, expanding source-code includes."""

import hashlib
import json
import re
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
RAW = "https://raw.githubusercontent.com/fastapi/fastapi"


def fetch_corpus():
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    lock_path = CORPUS / "snapshot.json"
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        if lock_path.exists():
            revision = json.loads(lock_path.read_text())["revision"]
        else:
            result = client.get("https://api.github.com/repos/fastapi/fastapi/commits/master")
            result.raise_for_status()
            revision = result.json()["sha"]
        fetched = []
        for item in manifest:
            response = client.get(f"{RAW}/{revision}/{item['path']}")
            response.raise_for_status()
            text = response.text
            # Official Markdown references Python source through custom include directives.
            # Inline complete files so the corpus contains runnable examples, not placeholders.
            includes = re.findall(
                r"\{\*\s*((?:\.\./)*docs_src/[^\s*]+)(?:\s+[^*]*?)?\s*\*\}",
                text,
            )
            for include_path in dict.fromkeys(includes):
                path = include_path[include_path.index("docs_src/") :]
                example = client.get(f"{RAW}/{revision}/{path}")
                example.raise_for_status()
                pattern = r"\{\*\s*" + re.escape(include_path) + r"(?:\s+[^*]*?)?\s*\*\}"
                replacement = "```python\n" + example.text.rstrip() + "\n```"
                text = re.sub(pattern, lambda _, value=replacement: value, text)
            # Strip presentation-only tab wrappers while retaining their content.
            text = re.sub(r"^/{3,}.*$", "", text, flags=re.MULTILINE)
            text = re.sub(r"\s*\{\s*#[^}]+\}", "", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
            output = CORPUS / item["filename"]
            output.write_text(text, encoding="utf-8")
            fetched.append({**item, "sha256": hashlib.sha256(text.encode()).hexdigest()})
            print(f"Fetched {item['filename']} ({len(text)} characters)")
        lock_path.write_text(
            json.dumps({"revision": revision, "documents": fetched}, indent=2) + "\n"
        )
        license_response = client.get(f"{RAW}/{revision}/LICENSE")
        license_response.raise_for_status()
        (CORPUS / "LICENSE.fastapi.txt").write_text(license_response.text)
        print(f"Pinned FastAPI revision: {revision}")


if __name__ == "__main__":
    fetch_corpus()
