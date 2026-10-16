from app.ingestion import Chunk
from app.store import VectorStore


def item(identifier, text, *, section="Section", title="Document"):
    return Chunk(identifier, "doc", text, title, "guide.html", section, 0)


class SearchResults:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k):
        self.calls.append(query)
        return self.results[query][:top_k]


def test_original_question_evidence_survives_a_drifting_rewrite():
    heading = item("title", "Document")
    unrelated = item("borrowers", "Three borrower examples")
    card = item("card", "A Negotiation Card is one screen a borrower can hold up to a lender.")
    scoring = item("scoring", "The app is evaluated on product quality")
    other = item("other", "Additional workflow information")
    store = SearchResults(
        {
            "original": [heading, unrelated, card, scoring],
            "rewritten": [heading, unrelated, other, scoring],
        }
    )
    hits = VectorStore.search_many(store, ["original", "rewritten"], 3)
    assert card in hits
    assert heading not in hits
    assert len(hits) <= 3
    assert len({hit.chunk_id for hit in hits}) == len(hits)


def test_same_original_and_rewrite_only_search_once():
    evidence = item("definition", "A concrete definition and an example")
    store = SearchResults({"same": [evidence]})
    assert VectorStore.search_many(store, ["same", "same"], 5) == [evidence]
    assert store.calls == ["same"]


def test_title_only_results_do_not_become_answer_evidence():
    store = SearchResults({"query": [item("title", "# Document")]})
    assert VectorStore.search_many(store, ["query"], 5) == []
