from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QueryRequest(StrictModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(StrictModel):
    citation: int = Field(ge=1)
    document_id: str
    chunk_id: str
    title: str
    source: str
    section: str
    excerpt: str


class TraceStep(StrictModel):
    node: str
    attempt: int
    detail: str


class QueryResponse(StrictModel):
    query_id: str
    answer: str
    sources: list[Source]
    status: Literal["answered", "insufficient_context"]
    retry_count: int
    retrieval_attempts: int
    trace: list[TraceStep]


class DocumentInfo(StrictModel):
    document_id: str
    title: str
    source: str
    chunk_count: int
    indexed_at: str


class IngestResponse(StrictModel):
    document: DocumentInfo
    duplicate: bool


class FeedbackRequest(StrictModel):
    query_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(StrictModel):
    feedback_id: str
    query_id: str
    rating: Literal["up", "down"]


class QueryAnalysis(StrictModel):
    query: str = Field(min_length=1, max_length=2000)
    query_type: Literal["conceptual", "how_to", "troubleshooting", "api_reference"]


class Grade(StrictModel):
    relevant: bool = Field(strict=True)
    reason: str = Field(min_length=1, max_length=500)


class AnswerDraft(StrictModel):
    answer: str = Field(min_length=1, max_length=16000)
    sufficient_context: bool = Field(strict=True)


class Groundedness(StrictModel):
    supported: bool = Field(strict=True)
    reason: str = Field(min_length=1, max_length=500)
