"""
FastAPI Application Entrypoint for Video Course Knowledge Engine.

Exposes:
- REST API for query retrieval and context synthesis
- Health monitoring and corpus indexing telemetry
- Static single-page application mounting
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure relative backend modules resolve across container runtimes
BACKEND_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rag_engine import ask, load_corpus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="PyTutor AI — Autonomous Video Knowledge Engine",
    description="Context-grounded question-answering service powered by vector embeddings and speech transcript retrieval.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="User inquiry text.")
    top_k: int = Field(default=4, ge=1, le=10, description="Maximum relevant chunks to retrieve.")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the difference between mutable and immutable types in Python?",
                "top_k": 4,
            }
        }


class SourceDocument(BaseModel):
    video_number: int
    title: str
    start: float
    end: float
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]


class HealthResponse(BaseModel):
    status: str
    version: str


class CorpusStatsResponse(BaseModel):
    total_chunks: int
    total_videos: int
    video_ids: List[int]


# ══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE HOOKS
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event() -> None:
    """Pre-warm corpus vectors on server launch to minimize first-request latency."""
    try:
        load_corpus()
        logger.info("Corpus successfully initialized and indexed.")
    except Exception as exc:
        logger.error("Failed to load corpus during startup: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# REST API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System Telemetry"])
def health_check() -> HealthResponse:
    """Service liveness probe."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/ask", response_model=QueryResponse, tags=["Retrieval-Augmented Generation"])
def ask_question(request: QueryRequest) -> QueryResponse:
    """Execute vector retrieval and synthesize response grounded in lecture context."""
    cleaned_query = request.question.strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inquiry text cannot be blank.",
        )

    try:
        result = ask(cleaned_query, top_k=request.top_k)
        return QueryResponse(**result)
    except Exception as exc:
        logger.exception("Error processing inquiry: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the answer. Please try again later.",
        ) from exc


@app.get("/corpus/stats", response_model=CorpusStatsResponse, tags=["System Telemetry"])
def get_corpus_statistics() -> CorpusStatsResponse:
    """Retrieve indexed video metadata statistics."""
    try:
        corpus = load_corpus()
        video_ids = sorted({c["video_number"] for c in corpus})
        return CorpusStatsResponse(
            total_chunks=len(corpus),
            total_videos=len(video_ids),
            video_ids=video_ids,
        )
    except Exception as exc:
        logger.exception("Failed to compute corpus statistics: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# STATIC SPA DELIVERY
# ══════════════════════════════════════════════════════════════════════════════

FRONTEND_DIR: Path = ROOT_DIR / "frontend"

@app.get("/", include_in_schema=False)
def serve_index() -> Any:
    """Deliver the client application bundle."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return {"status": "online", "message": "API active. Client bundle not installed."}

if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
