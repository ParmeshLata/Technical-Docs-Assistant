import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import Settings
from app.errors import ConfigurationError, InvalidDocument, ProviderError
from app.graph import build_graph, initial_state
from app.llm import JsonLLM
from app.schemas import (
    DocumentInfo,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.store import Repository, VectorStore


def create_app(settings: Settings | None = None, *, store=None, llm=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.store = store if store is not None else VectorStore(settings)
        application.state.llm = llm if llm is not None else JsonLLM(settings)
        application.state.repository = Repository(settings)
        application.state.graph = build_graph(
            settings,
            application.state.store,
            application.state.llm,
        )
        application.state.query_slots = asyncio.Semaphore(4)
        yield
        await application.state.llm.close()

    application = FastAPI(
        title="Technical Documentation Assistant",
        description="Corrective LangGraph RAG with source citations and bounded retrieval retries.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.exception_handler(ConfigurationError)
    async def configuration_error(_request: Request, exc: ConfigurationError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @application.exception_handler(ProviderError)
    async def provider_error(_request: Request, exc: ProviderError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @application.exception_handler(InvalidDocument)
    async def document_error(_request: Request, exc: InvalidDocument):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @application.get("/", include_in_schema=False)
    async def home():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @application.get("/health")
    async def health(request: Request):
        return {
            "status": "ok",
            "llm_configured": bool(settings.llm_api_key.get_secret_value()),
            "embedding_backend": settings.embedding_backend,
        }

    @application.post("/query", response_model=QueryResponse)
    async def query(body: QueryRequest, request: Request):
        state = request.app.state
        try:
            # Deadline includes queueing, retrieval, all model calls, and query persistence.
            async with asyncio.timeout(settings.query_timeout_seconds):
                async with state.query_slots:
                    if not await asyncio.to_thread(state.store.list_documents):
                        raise HTTPException(409, "No indexed documents. Run ingestion first.")
                    state.llm.ensure_configured()
                    result = await state.graph.ainvoke(
                        initial_state(body.question),
                        config={
                            "recursion_limit": 8 + 4 * settings.max_retries,
                        },
                    )
                    response = QueryResponse(
                        query_id=str(uuid4()),
                        answer=result["answer"],
                        sources=result["sources"],
                        status=result["status"],
                        retry_count=result["retry_count"],
                        retrieval_attempts=result["retrieval_attempts"],
                        trace=result["trace"],
                    )
                    await asyncio.to_thread(state.repository.save_query, body.question, response)
                    return response
        except TimeoutError as exc:
            raise HTTPException(
                504, "Query deadline exceeded. Try again or simplify the question."
            ) from exc

    @application.post("/ingest", response_model=IngestResponse, status_code=201)
    async def ingest(request: Request, file: Annotated[UploadFile, File()]):
        try:
            data = await file.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, "File exceeds MAX_UPLOAD_BYTES")
            # Filename is metadata only. Never write a user-supplied filesystem path.
            filename = (file.filename or "document.txt").replace("\\", "/").rsplit("/", 1)[-1]
            return await asyncio.to_thread(request.app.state.store.ingest, filename, data)
        finally:
            await file.close()

    @application.get("/documents", response_model=list[DocumentInfo])
    async def documents(request: Request):
        return await asyncio.to_thread(request.app.state.store.list_documents)

    @application.post("/feedback", response_model=FeedbackResponse, status_code=201)
    async def feedback(body: FeedbackRequest, request: Request):
        feedback_id = await asyncio.to_thread(
            request.app.state.repository.add_feedback,
            body.query_id,
            body.rating,
            body.comment,
        )
        if feedback_id is None:
            raise HTTPException(404, "Query ID not found")
        return FeedbackResponse(
            feedback_id=feedback_id,
            query_id=body.query_id,
            rating=body.rating,
        )

    return application


app = create_app()
