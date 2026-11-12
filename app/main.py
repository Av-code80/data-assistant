"""HTTP layer: FastAPI routes. They validate input, delegate to the services,
and translate domain errors into HTTP status codes.

Run (from data-assistant/):  uv run uvicorn app.main:app --reload
API docs:                    http://127.0.0.1:8000/docs
"""

import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .agent import SqlAgent
from .charts import suggest_chart
from .config import get_settings
from .knowledge import load_sections
from .llm import LLMRefusal, LLMService, LLMTruncated, LLMUnavailable
from .logging_config import configure_logging
from .retrieval import FastEmbedEmbedder, KnowledgeIndex
from .schemas import (AgentStepOut, AnswerRequest, AskResponse, Cell, InterpretRequest,
                      InterpretResponse, Source)
from .sql_tool import SqlTool

configure_logging()


@lru_cache
def get_llm_service() -> LLMService:
    """Dependency injection: one service (and one HTTP connection pool) for the
    whole app. Tests replace it with a fake."""
    return LLMService(get_settings())


@lru_cache
def get_knowledge_index() -> KnowledgeIndex:
    """Load the embedding model and index the knowledge base, once."""
    settings = get_settings()
    return KnowledgeIndex(load_sections(settings.knowledge_dir),
                          FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir))


@lru_cache
def get_sql_agent() -> SqlAgent:
    """One agent for the whole app: it reads the database schema once, at creation."""
    settings = get_settings()
    return SqlAgent(get_llm_service(), SqlTool(settings.analytics_db_path, settings.sql_max_rows,
                                               settings.sql_timeout_s))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the index at startup, so the first user does not wait for the model to load.
    # Skipped when tests override the dependency.
    if get_knowledge_index not in app.dependency_overrides:
        get_knowledge_index()
    yield


app = FastAPI(title="Data Assistant", version="0.3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins,
                   allow_methods=["*"], allow_headers=["*"])


# Domain errors -> HTTP status codes, in one place for every JSON route.
LLM_ERRORS = {
    LLMUnavailable: (503, "AI service temporarily unavailable, please retry."),
    LLMRefusal: (422, "This request cannot be processed."),
    LLMTruncated: (502, "Incomplete AI response."),
}


async def llm_error_handler(request, error):
    status_code, detail = LLM_ERRORS[type(error)]
    return JSONResponse({"detail": detail}, status_code=status_code)


for error_type in LLM_ERRORS:
    app.add_exception_handler(error_type, llm_error_handler)


def to_cell(value) -> Cell:
    """Database values -> JSON values. Decimal and dates have no JSON type."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def to_sources(hits) -> list[Source]:
    return [Source(id=h.section.id, title=h.section.title, file=h.section.source, score=h.vector)
            for h in hits]


@app.get("/health")
def health():
    """Health probe: Cloud Run or Kubernetes call it to check the app is alive."""
    return {"status": "ok", "version": app.version}


@app.get("/knowledge/search", response_model=list[Source])
def search_knowledge(q: str = Query(min_length=2, max_length=1_000),
                     index: KnowledgeIndex = Depends(get_knowledge_index)):
    """What the RAG would retrieve for a question: for debugging and transparency."""
    settings = get_settings()
    return to_sources(index.search(q, settings.retrieval_top_k,
                                   settings.retrieval_min_vector_score))


@app.post("/questions/interpret", response_model=InterpretResponse)
async def interpret_question(request: InterpretRequest,
                             service: LLMService = Depends(get_llm_service),
                             index: KnowledgeIndex = Depends(get_knowledge_index)):
    """Step 1 + 2: retrieve the company's definitions, then understand the question."""
    settings = get_settings()
    hits = index.search(request.question, settings.retrieval_top_k,
                        settings.retrieval_min_vector_score)
    result = await service.interpret_question(request.question,
                                              context=[hit.section for hit in hits])
    return InterpretResponse(status=result.status, interpretation=result.interpretation,
                             warnings=result.warnings, sources=to_sources(hits),
                             usage=result.usage)


@app.post("/questions/ask", response_model=AskResponse)
async def ask_question(request: InterpretRequest,
                       service: LLMService = Depends(get_llm_service),
                       index: KnowledgeIndex = Depends(get_knowledge_index),
                       agent: SqlAgent = Depends(get_sql_agent)):
    """Steps 1 to 3 in one call: retrieve, interpret, then let the agent write and run the SQL.
    The agent only runs when the interpretation is `ready`."""
    settings = get_settings()
    hits = index.search(request.question, settings.retrieval_top_k,
                        settings.retrieval_min_vector_score)
    context = [hit.section for hit in hits]
    interpreted = await service.interpret_question(request.question, context=context)
    response = AskResponse(status=interpreted.status if interpreted.status != "ready" else "failed",
                           interpretation=interpreted.interpretation,
                           warnings=interpreted.warnings, sources=to_sources(hits),
                           usage=[interpreted.usage], total_cost_usd=interpreted.usage.cost_usd)
    if interpreted.status != "ready":
        return response      # a clarifying question or a refusal costs no SQL call

    outcome = await agent.answer(request.question, interpreted.interpretation, context)
    response.status = outcome.status
    response.steps = [AgentStepOut(**asdict(step)) for step in outcome.steps]
    response.reason = outcome.reason
    response.usage.append(outcome.usage)
    response.total_cost_usd = round(interpreted.usage.cost_usd + outcome.usage.cost_usd, 6)
    if outcome.result:
        response.sql = outcome.result.sql
        response.columns = outcome.result.columns
        response.rows = [[to_cell(value) for value in row] for row in outcome.result.rows]
        response.truncated = outcome.result.truncated
        response.chart = suggest_chart(outcome.result.columns, outcome.result.rows)
    return response


@app.post("/answers/stream")
async def stream_answer(request: AnswerRequest, service: LLMService = Depends(get_llm_service)):
    """Last step: explain the result table, as Server-Sent Events.
    In React, read the body with fetch + ReadableStream."""
    async def events():
        try:
            async for chunk in service.stream_answer(request):
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except (LLMUnavailable, LLMRefusal, LLMTruncated) as error:
            # Headers are already sent, so the error is reported inside the stream.
            yield f"event: error\ndata: {json.dumps({'type': type(error).__name__})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
