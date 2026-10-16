"""Deterministic test doubles. Never used as an application runtime fallback."""

import hashlib
import math
import re

from app.schemas import AnswerDraft, Grade, Groundedness, QueryAnalysis


def test_embeddings(texts):
    result = []
    for text in texts:
        vector = [0.0] * 384
        for word in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 384
            vector[bucket] += 1
        length = math.sqrt(sum(value * value for value in vector)) or 1
        result.append([value / length for value in vector])
    return result


class FakeLLM:
    def __init__(self, grades=None, *, bad_citations=False, supported=True, sufficient=True):
        self.grades = iter(grades) if grades is not None else None
        self.bad_citations = bad_citations
        self.supported = supported
        self.sufficient = sufficient
        self.calls = []

    def ensure_configured(self):
        pass

    async def close(self):
        pass

    async def ask(self, schema, system, payload):
        self.calls.append((schema.__name__, payload))
        if schema is QueryAnalysis:
            retry = len(payload.get("previous_queries", []))
            return QueryAnalysis(query=f"{payload['question']} search {retry}", query_type="how_to")
        if schema is Grade:
            relevant = next(self.grades) if self.grades is not None else True
            return Grade(relevant=relevant, reason="Controlled relevance decision")
        if schema is AnswerDraft:
            return AnswerDraft(
                answer=(
                    "Use a Pydantic model for the request body "
                    + ("[99]." if self.bad_citations else "[1].")
                ),
                sufficient_context=self.sufficient,
            )
        if schema is Groundedness:
            return Groundedness(supported=self.supported, reason="Controlled groundedness decision")
        raise AssertionError(schema)


class ScriptedStore:
    def __init__(self, results):
        self.results = iter(results)
        self.queries = []
        self.query_groups = []

    def search(self, query, top_k):
        self.queries.append(query)
        return next(self.results)

    def search_many(self, queries, top_k):
        self.query_groups.append(queries)
        self.queries.append(queries[-1])
        return next(self.results)
