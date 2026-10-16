# Validation record

Development checks were run in this environment on 2026-10-15 with Python 3.12.14 on Linux.
The exact direct dependency versions are recorded in `requirements.txt`.

## Completed

| Check | Result |
|---|---|
| `python -m pytest -q` | 41 passed |
| `python -m ruff check .` | Passed |
| `python -m ruff format --check .` | Passed |
| Installed dependency compatibility check | Passed |
| Corpus download and code-include expansion | Five pages fetched from a pinned FastAPI commit |
| Corpus integrity | All five processed files matched `snapshot.json` SHA-256 values |
| Actual local MiniLM embedding download/inference | Passed |
| Fresh persistent Chroma ingestion | Five documents, 98 chunks |
| Actual embedding retrieval checks | Expected documentation source appeared in top five for all five supplied answerable questions |
| Real Uvicorn process startup and `/health` | Passed |
| Local interface JavaScript syntax | Passed |
| Local embedding initialization telemetry opt-out | Unit test verifies opt-out occurs before model initialization |

The automated tests run the real graph, API, vector database, and feedback database. They substitute
controlled chat responses and deterministic embeddings where needed to test routing and errors
without network calls. Separately, `scripts/check_retrieval.py` was run with real MiniLM embeddings
and the included corpus. Its five retrieval checks do not establish general retrieval accuracy.

## Requires a real LLM key

No chat-provider API key was configured in the development environment. Therefore, real-provider
query rewriting, relevance judgments, answer quality, provider-specific JSON behavior, and the optional
groundedness model call have **not** been evaluated end to end against a hosted model.

After setting `LLM_API_KEY`, ingest the corpus, start the API, and run:

```bash
python -m scripts.evaluate
```

Manually inspect the printed answers, citations, and unsupported-question responses. Matching an
expected status and source URL is an integration contract check, not proof of factual correctness.

## Portability limits

The local HTML interface is served successfully and its JavaScript syntax was checked. Visual
browser interaction was not verified: a Chromium executable was unavailable and its download failed
in this environment. The main FastAPI behavior is covered by the API tests above.

Windows and macOS installation were not exercised here. The README includes Windows activation and
configuration commands, and the project uses cross-platform Python path handling. A fresh environment
still needs access to the package index, the initial embedding download, and the selected chat API.

GitHub publication and hosted deployment are not included in this local validation record.

## Retrieval and grading update - 2026-10-16

A reported failure involved a correctly indexed HTML document whose Negotiation Card explanation
was present in direct vector-search results, while the graph repeatedly rejected all three
candidates. The old trace did not reveal the rewritten query or the grader reasons, so the exact
live-model cause could not be established from that trace alone.

The update adds original-question search alongside the rewritten query, excludes title-only hits,
interleaves/deduplicates results within TOP_K, preserves document-specific names in analysis prompts,
clarifies partial-evidence relevance grading for uploaded briefs, and exposes candidate previews and
grading reasons. The LLM still decides relevance; no keyword override forces a chunk through.

Validation after the update:

- All 45 automated tests passed, including the existing API and workflow checks.
- Ruff lint and formatting checks passed.
- Actual MiniLM/Chroma indexing of the five FastAPI documents plus the user-provided Lokta HTML
  selected the Negotiation Card explanation at both TOP_K=3 and TOP_K=5, even when paired with a
  deliberately drifting technical rewrite.
- Dependencies were installed using pyproject.toml compatibility ranges for this update;
  FastAPI 0.143.2, LangGraph 1.2.15, Chroma 1.5.9, OpenAI client 2.54.0, and Ruff 0.17.1 were used.
- No chat-provider key was available. Live Groq grading and generation still require verification.

The index format, embedding configuration, and chunking settings did not change; no re-ingestion
is required for this update. retrieval_attempts now counts graph retrieval rounds, not individual
vector-search invocations. Each round searches at most two distinct query variants.

## Generation and citation update - 2026-10-16

A subsequent live trace showed three accepted chunks followed by the old combined message
"Insufficient context or invalid citations". That trace established the rejection stage but did not
show which of those two conditions caused the rejection.

The internal answer schema now asks for answer text and a sufficient-context flag, with numeric
citations in the text. The application maps valid numbers to actual source IDs; the model no longer
has to repeat chunk IDs in a separate JSON field. Invalid or missing references trigger at most one
repair call. Repeatedly invalid references still produce a fallback with no sources. The trace now
separates the model's insufficient-context decision from missing/unknown references and repair
failure. Public API response fields have not changed.

All 50 automated tests passed after this change, including new checks for numeric source mapping,
a successful bounded repair, repeated missing references, insufficient-context tracing, and multiple
cited sources. Lint/format checks passed. No hosted LLM key was available, so real Groq generation
remains unverified. This update addresses a validation failure mode; it does not establish the exact
cause of the user's previous combined-error trace.
