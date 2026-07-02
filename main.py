"""
FastAPI application for the SHL Assessment Recommender.

Endpoints:
    GET  /health  — Returns {"status": "ok"} (liveness check)
    POST /chat    — Accepts a ChatRequest, returns a ChatResponse

The catalog and FAISS index are initialized once at startup via the
FastAPI lifespan context manager.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from agent import get_agent_response
from models import ChatRequest, ChatResponse
from retrieval import init_catalog

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the catalog and FAISS index at startup; clean up on shutdown.

    This ensures the expensive model loading and index building happens
    exactly once before the first request is served.
    """
    logger.info("Starting SHL Assessment Recommender — initializing catalog …")
    start = time.perf_counter()
    init_catalog()
    logger.info("Catalog ready in %.2fs", time.perf_counter() - start)
    yield
    logger.info("Shutting down SHL Assessment Recommender.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "A conversational agent that helps hiring managers and recruiters "
        "find the right SHL assessments for their roles."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    Returns:
        A JSON object with {"status": "ok"} and HTTP 200.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Process a conversational turn.

    The client sends the full conversation history in every request
    (the service is stateless). The agent retrieves relevant catalog
    items, calls Gemini, and returns a structured response.

    Args:
        request: A ChatRequest containing the full message history.

    Returns:
        A ChatResponse with the agent's reply, recommendations, and
        end_of_conversation flag.
    """
    start = time.perf_counter()
    logger.info(
        "POST /chat — %d message(s) in history",
        len(request.messages),
    )

    response = get_agent_response(request.messages)

    elapsed = time.perf_counter() - start
    logger.info(
        "POST /chat completed in %.2fs — %d recommendation(s), eoc=%s",
        elapsed,
        len(response.recommendations),
        response.end_of_conversation,
    )
    return response
