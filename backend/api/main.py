"""
FastAPI server exposing the full Enterprise Intelligence Platform pipeline
(Router -> SQL/RAG/ML -> Synthesis -> Verification).

Production additions in this version:
  - Optional API key authentication (set API_KEY in .env to require it;
    leave it unset to keep running without auth, for local development).
  - Structured request logging (question, duration, confidence, status).
  - A global rate limit per API key / IP (simple in-memory limiter — good
    enough for a single-instance deployment, not for a multi-server one).

Run with:
    uvicorn backend.api.main:app --reload

Then either:
  - open http://localhost:8000/docs for an interactive test UI (Swagger), or
  - open frontend/index.html in your browser for the chat UI.
"""

import sys
import os
import time
from collections import defaultdict, deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.orchestrator import ask as run_pipeline
from backend.logging_config import get_logger

load_dotenv()
logger = get_logger("api")

API_KEY = os.getenv("API_KEY")  # if unset, auth is disabled (local dev mode)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))

app = FastAPI(title="Enterprise Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for local dev; restrict to known origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# simple in-memory rate limiter: {client_key: deque[timestamps]}
_request_log = defaultdict(deque)


def check_rate_limit(client_key: str):
    now = time.time()
    window = _request_log[client_key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    window.append(now)


def check_api_key(x_api_key: str = Header(default=None)):
    if API_KEY is None:
        return  # auth disabled — local dev mode
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class AskRequest(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "ok", "message": "Enterprise Intelligence Platform API is running"}


@app.post("/ask")
def ask(req: AskRequest, request: Request, x_api_key: str = Header(default=None)):
    check_api_key(x_api_key)
    client_key = x_api_key or request.client.host
    check_rate_limit(client_key)

    start = time.time()
    logger.info(f"Question received: {req.question!r}")

    try:
        result = run_pipeline(req.question)
    except Exception as e:
        logger.error(f"Pipeline failed for question {req.question!r}: {e}")
        raise HTTPException(status_code=500, detail="Internal error processing the question.")

    duration = round(time.time() - start, 2)
    confidence = result["verification"]["confidence"]
    logger.info(
        f"Answered in {duration}s | sources={result['sources']} | confidence={confidence}"
    )

    return {
        "question": result["question"],
        "sources": result["sources"],
        "answer": result["answer"],
        "confidence": confidence,
        "warnings": result["verification"]["warnings"],
        "response_time_seconds": duration,
    }
