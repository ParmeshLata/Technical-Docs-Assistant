import pytest

from app.config import Settings
from app.graph import FALLBACK, build_graph, citation_numbers, initial_state
from app.ingestion import Chunk
from app.schemas import AnswerDraft
from tests.fakes import FakeLLM, ScriptedStore


def chunk(identifier="doc:0", text="A Pydantic model defines a request body."):
    return Chunk(identifier, "doc", text, "Request body", "body.md", "Models", 0)


def test_code_indices_are_not_treated_as_citations():
    answer = "Use this example [1].\n```python\nvalue = items[0]\n```\nTry `items[9]` [2]."
    assert citation_numbers(answer) == {1, 2}


async def test_relevant_chunks_generate_without_retry():
    model = FakeLLM()
    result = await build_graph(Settings(), ScriptedStore([[chunk()]]), model).ainvoke(
        initial_state("How do request bodies work?")
    )
    assert result["status"] == "answered"
    assert result["retrieval_attempts"] == 1
    assert result["retry_count"] == 0
    assert result["sources"][0].chunk_id == "doc:0"


async def test_mixed_results_filter_irrelevant_evidence():
    model = FakeLLM(grades=[False, True])
    result = await build_graph(
        Settings(),
        ScriptedStore(
            [
                [chunk("bad:0", "Unrelated database material"), chunk("good:0")],
            ]
        ),
        model,
    ).ainvoke(initial_state("Request bodies?"))
    generation = next(payload for name, payload in model.calls if name == "AnswerDraft")
    assert [c["text"] for c in generation["evidence"]] == [chunk("good:0").text]
    assert [c.chunk_id for c in result["relevant_chunks"]] == ["good:0"]
    assert result["status"] == "answered"


async def test_retry_replaces_evidence_and_preserves_original_question():
    model = FakeLLM(grades=[False, True])
    store = ScriptedStore([[chunk("old:0")], [chunk("new:0")]])
    result = await build_graph(Settings(), store, model).ainvoke(initial_state("Original question"))
    assert result["retry_count"] == 1
    assert result["retrieval_attempts"] == 2
    assert [c.chunk_id for c in result["relevant_chunks"]] == ["new:0"]
    assert set(result["grades"]) == {"new:0"}
    assert store.queries[0] != store.queries[1]
    assert all(payload["question"] == "Original question" for _, payload in model.calls)
    assert all(group[0] == "Original question" for group in store.query_groups)


async def test_trace_exposes_actual_search_candidates_and_grading_reason():
    model = FakeLLM(grades=[False])
    result = await build_graph(
        Settings(max_retries=0),
        ScriptedStore([[chunk("trace:0")]]),
        model,
    ).ainvoke(initial_state("Request body?"))
    steps = {step["node"]: step["detail"] for step in result["trace"]}
    assert "retrieval query:" in steps["query_analysis"]
    assert "body.md" in steps["retrieval"]
    assert "Pydantic model" in steps["retrieval"]
    assert "irrelevant (Controlled relevance decision)" in steps["document_grading"]


@pytest.mark.parametrize("max_retries", [0, 1, 2, 3])
async def test_retry_limit_terminates_without_generation(max_retries):
    attempts = max_retries + 1
    model = FakeLLM(grades=[False] * attempts)
    store = ScriptedStore([[chunk()]] * attempts)
    result = await build_graph(Settings(max_retries=max_retries), store, model).ainvoke(
        initial_state("Unknown topic"),
        config={"recursion_limit": 8 + 4 * max_retries},
    )
    assert len(store.queries) == attempts
    assert result["retry_count"] == max_retries
    assert result["answer"] == FALLBACK
    assert result["sources"] == []
    assert not any(name == "AnswerDraft" for name, _ in model.calls)


async def test_empty_retrieval_does_not_call_grader():
    model = FakeLLM()
    result = await build_graph(Settings(max_retries=0), ScriptedStore([[]]), model).ainvoke(
        initial_state("No evidence"),
    )
    assert result["status"] == "insufficient_context"
    assert [name for name, _ in model.calls] == ["QueryAnalysis"]


@pytest.mark.parametrize("options", [{"bad_citations": True}, {"sufficient": False}])
async def test_bad_citations_or_insufficient_answer_fail_closed(options):
    result = await build_graph(Settings(), ScriptedStore([[chunk()]]), FakeLLM(**options)).ainvoke(
        initial_state("Request body?"),
    )
    assert result["status"] == "insufficient_context"
    assert result["sources"] == []


@pytest.mark.parametrize("supported", [True, False])
async def test_optional_groundedness_check(supported):
    result = await build_graph(
        Settings(check_groundedness=True),
        ScriptedStore([[chunk()]]),
        FakeLLM(supported=supported),
    ).ainvoke(initial_state("Request body?"))
    assert result["status"] == ("answered" if supported else "insufficient_context")
    assert result["trace"][-1]["node"] == "groundedness"


async def test_numeric_references_map_to_real_ids_without_model_supplied_ids():
    result = await build_graph(
        Settings(),
        ScriptedStore([[chunk("real-document-id:7")]]),
        FakeLLM(),
    ).ainvoke(initial_state("Request body?"))
    assert result["status"] == "answered"
    assert result["sources"][0].chunk_id == "real-document-id:7"
    assert set(AnswerDraft.model_fields) == {"answer", "sufficient_context"}


async def test_one_citation_repair_can_recover_a_supported_answer():
    class RepairLLM(FakeLLM):
        async def ask(self, schema, system, payload):
            if schema is AnswerDraft and "draft_answer" in payload:
                self.calls.append((schema.__name__, payload))
                return AnswerDraft(answer="Use a Pydantic model [1].", sufficient_context=True)
            return await super().ask(schema, system, payload)

    model = RepairLLM(bad_citations=True)
    result = await build_graph(Settings(), ScriptedStore([[chunk()]]), model).ainvoke(
        initial_state("Request body?"),
    )
    assert result["status"] == "answered"
    assert sum(name == "AnswerDraft" for name, _ in model.calls) == 2
    assert "after one citation-format repair" in result["trace"][-1]["detail"]


async def test_missing_citations_are_not_silently_invented():
    class UncitedLLM(FakeLLM):
        async def ask(self, schema, system, payload):
            if schema is AnswerDraft:
                self.calls.append((schema.__name__, payload))
                return AnswerDraft(answer="An uncited statement.", sufficient_context=True)
            return await super().ask(schema, system, payload)

    model = UncitedLLM()
    result = await build_graph(Settings(), ScriptedStore([[chunk()]]), model).ainvoke(
        initial_state("Request body?"),
    )
    assert result["status"] == "insufficient_context"
    assert result["sources"] == []
    assert sum(name == "AnswerDraft" for name, _ in model.calls) == 2
    assert "No inline source citations" in result["trace"][-1]["detail"]


async def test_insufficient_context_is_distinguished_from_invalid_references():
    model = FakeLLM(sufficient=False)
    result = await build_graph(Settings(), ScriptedStore([[chunk()]]), model).ainvoke(
        initial_state("Request body?"),
    )
    assert "Model reported insufficient context" in result["trace"][-1]["detail"]
    assert sum(name == "AnswerDraft" for name, _ in model.calls) == 1


async def test_multiple_citations_map_only_to_the_cited_evidence():
    class MultiSourceLLM(FakeLLM):
        async def ask(self, schema, system, payload):
            if schema is AnswerDraft:
                return AnswerDraft(
                    answer="First fact [1]. Third fact [3].", sufficient_context=True
                )
            return await super().ask(schema, system, payload)

    result = await build_graph(
        Settings(),
        ScriptedStore([[chunk("a"), chunk("b"), chunk("c")]]),
        MultiSourceLLM(),
    ).ainvoke(initial_state("Request body?"))
    assert result["status"] == "answered"
    assert [(source.citation, source.chunk_id) for source in result["sources"]] == [
        (1, "a"),
        (3, "c"),
    ]
