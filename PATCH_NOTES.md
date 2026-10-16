# Retrieval and generation update (v2)

## Apply to your existing project

1. Stop Uvicorn with Ctrl+C.
2. Extract this ZIP into the existing project root, the folder containing `app`, `tests`, and
   `pyproject.toml`. Replace the files when prompted. All three runtime files, `app/graph.py`, `app/store.py`, and `app/schemas.py`, must
   be replaced together. The included tests and documentation describe the updated behavior.
3. Keep your existing `.env`, API key, uploaded documents, and runtime database. They are not
   included in this update. No dependency changes or re-ingestion are required.
4. For the next test, set `TOP_K=5` in your own `.env` and leave `CHECK_GROUNDEDNESS=false`.
5. With your environment activated, restart:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Test with complete questions, for example:

- `What is the Negotiation Card described in the Borrower Copilot challenge?`
- `How many points are allocated to domain reasoning, and what do the judges assess?`

The expected evidence describes a screen the borrower can show to a lender, with a profile-based
fair-rate band and its reasoning to support negotiation. Rates quoted in that passage are examples,
not a calculated rate for a particular borrower.

If the application still falls back, copy the new workflow trace. It now includes the retrieval
query, source and section previews for each candidate, and the relevance reason for each chunk.

## What changed

- Both the original question and rewritten query are searched, so the original document terms
  remain represented if the rewrite drifts.
- Results consisting only of a title/heading no longer occupy the grading budget.
- Results are interleaved and deduplicated, with at most TOP_K chunks passed to the LLM grader.
- The analysis prompt preserves names/acronyms and allows uploaded business briefs.
- The grading prompt explicitly accepts useful definitions, descriptions, examples, and partial
  evidence. It still rejects unrelated content; the grader is not bypassed.
- Workflow traces expose candidates and short grading reasons.

## Generation changes in v2

- Internal model output now contains answer text and a sufficient-context flag. Source IDs are
  derived by the application from validated inline numbers such as [1]. The public API is unchanged.
- Missing or unknown citation numbers trigger one bounded repair request. The application never
  invents a reference to make an answer pass. Repeated validation failure still returns a fallback.
- The trace distinguishes a model decision of insufficient context from missing/unknown citations.
- This adds at most one model call when citation repair is needed. Normal answers use no extra call.
- The original retrieval/grading updates are included, so this package can replace the earlier patch.

## Validation

50 automated tests passed, plus lint and formatting checks. Real MiniLM retrieval selected the
Negotiation Card evidence from the Lokta HTML at TOP_K=3 and TOP_K=5 in a six-document corpus.
Actual Groq grading and answer generation have not been tested here because no API key was provided.

The package contains only the changed source files, related tests, README, validation record,
and these instructions. It is an update for the existing project, not a standalone application.
