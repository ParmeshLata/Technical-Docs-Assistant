"""Corrective RAG StateGraph with replace-on-retrieval state and bounded retries."""

import asyncio
import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.ingestion import Chunk
from app.schemas import AnswerDraft, Grade, Groundedness, QueryAnalysis, Source

FALLBACK = (
    "I don't have enough relevant information in the indexed documents to answer this "
    "question reliably. Please upload documentation covering the topic or make your "
    "question more specific."
)


class RAGState(TypedDict):
    original_question: str
    retrieval_query: str
    query_type: str
    previous_queries: list[str]
    retrieved_chunks: list[Chunk]
    grades: dict[str, Grade]
    relevant_chunks: list[Chunk]
    retry_count: int
    retrieval_attempts: int
    answer: str
    sources: list[Source]
    status: Literal["answered", "insufficient_context"]
    trace: list[dict]


def initial_state(question: str) -> RAGState:
    return {
        "original_question": question,
        "retrieval_query": question,
        "query_type": "conceptual",
        "previous_queries": [],
        "retrieved_chunks": [],
        "grades": {},
        "relevant_chunks": [],
        "retry_count": 0,
        "retrieval_attempts": 0,
        "answer": "",
        "sources": [],
        "status": "insufficient_context",
        "trace": [],
    }


def trace(state: RAGState, node: str, detail: str) -> list[dict]:
    return [
        *state["trace"],
        {
            "node": node,
            "attempt": state["retrieval_attempts"],
            "detail": detail,
        },
    ]


def citation_numbers(answer: str) -> set[int]:
    # Python indexing inside examples (items[0]) must not be mistaken for a source citation.
    prose = re.sub(r"```.*?```|~~~.*?~~~", "", answer, flags=re.DOTALL)
    prose = re.sub(r"`[^`\n]*`", "", prose)
    return {int(number) for number in re.findall(r"(?<!\w)\[(\d+)\]", prose)}


def build_graph(settings: Settings, store, llm):
    grading_slots = asyncio.Semaphore(3)

    async def analyze(state: RAGState):
        analysis = await llm.ask(
            QueryAnalysis,
            "Prepare a precise vector-search query for the uploaded documents. Preserve the "
            "question's intent and its named terms, document names, and acronyms exactly. "
            "Documents may contain technical guides, requirements, or business briefs. Do not "
            "turn a business concept into a programming question or invent APIs or facts. "
            "Classify the query type.",
            {"question": state["original_question"]},
        )
        return {
            "retrieval_query": analysis.query,
            "query_type": analysis.query_type,
            "trace": trace(
                state,
                "query_analysis",
                (f"Type: {analysis.query_type}; retrieval query: {analysis.query}"),
            ),
        }

    async def retrieve(state: RAGState):
        queries = list(dict.fromkeys([state["original_question"], state["retrieval_query"]]))
        chunks = await asyncio.to_thread(store.search_many, queries, settings.top_k)
        attempts = state["retrieval_attempts"] + 1
        evidence = "; ".join(
            f"{i}: {c.source} | {c.section} | {' '.join(c.text.split())[:160]}"
            for i, c in enumerate(chunks, 1)
        )
        # These lists/maps intentionally have no append reducers: old evidence is discarded.
        return {
            "retrieved_chunks": chunks,
            "relevant_chunks": [],
            "grades": {},
            "retrieval_attempts": attempts,
            "previous_queries": [*state["previous_queries"], state["retrieval_query"]],
            "trace": [
                *state["trace"],
                {
                    "node": "retrieval",
                    "attempt": attempts,
                    "detail": f"Retrieved {len(chunks)} chunks. Candidates: {evidence}",
                },
            ],
        }

    async def grade(state: RAGState):
        async def grade_one(chunk: Chunk):
            async with grading_slots:
                return await llm.ask(
                    Grade,
                    "Assess this chunk against the original question using its actual content. "
                    "Set relevant=true if it contains a definition, description, example, fact, "
                    "or requirement that helps answer any part of the question. Partial evidence "
                    "is enough: do not require one chunk to answer the entire question. Documents "
                    "can be technical guides or business/project briefs; code is not required. "
                    "Set relevant=false for merely shared generic words or genuinely unrelated "
                    "content. Give a short evidence-based reason; do not answer the question.",
                    {
                        "question": state["original_question"],
                        "chunk": chunk.text,
                        "section": chunk.section,
                        "title": chunk.title,
                        "source": chunk.source,
                    },
                )

        decisions = await asyncio.gather(*(grade_one(c) for c in state["retrieved_chunks"]))
        grades = {
            c.chunk_id: decision
            for c, decision in zip(
                state["retrieved_chunks"],
                decisions,
                strict=True,
            )
        }
        relevant = [c for c in state["retrieved_chunks"] if grades[c.chunk_id].relevant]
        return {
            "grades": grades,
            "relevant_chunks": relevant,
            "trace": trace(
                state,
                "document_grading",
                (
                    f"Accepted {len(relevant)} of {len(decisions)} chunks. "
                    + "; ".join(
                        f"{i}: {'relevant' if decision.relevant else 'irrelevant'} "
                        f"({decision.reason[:220]})"
                        for i, decision in enumerate(decisions, 1)
                    )
                ),
            ),
        }

    def route(state: RAGState) -> Literal["generate", "rewrite", "fallback"]:
        if state["relevant_chunks"]:
            return "generate"
        if state["retry_count"] < settings.max_retries:
            return "rewrite"
        return "fallback"

    async def rewrite(state: RAGState):
        analysis = await llm.ask(
            QueryAnalysis,
            "The previous searches found no relevant evidence. Produce a different retrieval "
            "query: use alternative terms, simplify, or remove unsupported assumptions. "
            "Preserve the original intent. Do not answer the question.",
            {"question": state["original_question"], "previous_queries": state["previous_queries"]},
        )
        return {
            "retrieval_query": analysis.query,
            "retry_count": state["retry_count"] + 1,
            "trace": trace(
                state,
                "query_rewrite",
                (f"Rewrote query after irrelevant results: {analysis.query}"),
            ),
        }

    async def generate(state: RAGState):
        evidence = [
            {
                "citation": i,
                "text": c.text,
                "source": c.source,
                "section": c.section,
            }
            for i, c in enumerate(state["relevant_chunks"], 1)
        ]
        prompt = (
            "Answer the original question using ONLY the supplied evidence. Give clear, "
            "practical explanations and code when supported. Cite factual claims inline with "
            "the provided numbers, e.g. [1]. Use only single-number citations, not ranges. "
            "The application maps citation numbers to source IDs; do not output chunk IDs. "
            "If the question is a fragment clearly naming a heading, explain the matching "
            "section using the evidence. Set sufficient_context=true when the evidence "
            "supports a reliable answer; state any limits explicitly. Otherwise set "
            "sufficient_context=false and do not guess."
        )
        draft = await llm.ask(
            AnswerDraft,
            prompt,
            {"question": state["original_question"], "evidence": evidence},
        )
        known = set(range(1, len(evidence) + 1))

        def failure(detail: str):
            return {
                "answer": FALLBACK,
                "sources": [],
                "status": "insufficient_context",
                "trace": trace(state, "generation", detail),
            }

        def citation_problem(answer: str) -> str | None:
            numbers = citation_numbers(answer)
            if not numbers:
                return "No inline source citations were returned"
            unknown = numbers - known
            if unknown:
                return f"Unknown source citation numbers: {sorted(unknown)}"
            return None

        if not draft.sufficient_context:
            return failure(
                f"Model reported insufficient context after {len(evidence)} chunks passed grading"
            )
        problem = citation_problem(draft.answer)
        repaired = problem is not None
        if problem:
            # One bounded repair. Never append a citation automatically or accept invented sources.
            draft = await llm.ask(
                AnswerDraft,
                prompt + " Fix the draft's source references using the supplied evidence. "
                "Cite a source only when it actually supports the associated claim. "
                "Use the provided numeric references in square brackets. Do not invent sources. "
                "If the evidence cannot support an answer, set sufficient_context=false.",
                {
                    "question": state["original_question"],
                    "evidence": evidence,
                    "draft_answer": draft.answer,
                    "citation_issue": problem,
                    "allowed_citations": sorted(known),
                },
            )
            if not draft.sufficient_context:
                return failure("Model reported insufficient context during citation repair")
            problem = citation_problem(draft.answer)
            if problem:
                return failure(f"Citation validation failed after one repair: {problem}")
        inline = citation_numbers(draft.answer)
        sources = [
            Source(
                citation=i,
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                title=c.title,
                source=c.source,
                section=c.section,
                excerpt=c.text[:600],
            )
            for i, c in enumerate(state["relevant_chunks"], 1)
            if i in inline
        ]
        return {
            "answer": draft.answer,
            "sources": sources,
            "status": "answered",
            "trace": trace(
                state,
                "generation",
                (
                    f"Generated answer citing {len(sources)} chunks"
                    + (" after one citation-format repair" if repaired else "")
                ),
            ),
        }

    async def groundedness(state: RAGState):
        if state["status"] != "answered":
            return {}
        cited = {source.chunk_id for source in state["sources"]}
        verdict = await llm.ask(
            Groundedness,
            "Verify that every factual claim and code example in the answer is supported by "
            "the evidence. Ignore citation syntax; assess actual content. Return supported=false "
            "if any material claim is invented or contradicts the evidence.",
            {
                "question": state["original_question"],
                "answer": state["answer"],
                "evidence": [c.text for c in state["relevant_chunks"] if c.chunk_id in cited],
            },
        )
        result = {
            "trace": trace(
                state,
                "groundedness",
                ("Supported" if verdict.supported else "Rejected unsupported answer"),
            )
        }
        if not verdict.supported:
            result.update(answer=FALLBACK, sources=[], status="insufficient_context")
        return result

    async def fallback(state: RAGState):
        return {
            "answer": FALLBACK,
            "sources": [],
            "status": "insufficient_context",
            "trace": trace(state, "fallback", "No relevant evidence; retry limit reached"),
        }

    graph = StateGraph(RAGState)
    for name, function in (
        ("query_analysis", analyze),
        ("retrieval", retrieve),
        ("document_grading", grade),
        ("rewrite", rewrite),
        ("generate", generate),
        ("fallback", fallback),
    ):
        graph.add_node(name, function)
    graph.add_edge(START, "query_analysis")
    graph.add_edge("query_analysis", "retrieval")
    graph.add_edge("retrieval", "document_grading")
    graph.add_conditional_edges("document_grading", route)
    graph.add_edge("rewrite", "retrieval")
    graph.add_edge("fallback", END)
    if settings.check_groundedness:
        graph.add_node("groundedness", groundedness)
        graph.add_edge("generate", "groundedness")
        graph.add_edge("groundedness", END)
    else:
        graph.add_edge("generate", END)
    return graph.compile()
