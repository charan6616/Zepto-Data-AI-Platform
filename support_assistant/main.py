"""FastAPI wrapper around the LangGraph support assistant.

Run locally:  MOCK_LLM=1 uvicorn main:app --host 0.0.0.0 --port 7860
(Windows PowerShell:  $env:MOCK_LLM="1"; uvicorn main:app --host 0.0.0.0 --port 7860)
MOCK_LLM left at its default (unset or 1) = graded, fully offline mock baseline.
"""

import os

from fastapi import FastAPI

from app.graph import handle_query
from app.llm import MOCK_LLM
from app.schema import AnswerResponse, AskRequest

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0",
    description=(
        "Deterministic offline RAG assistant for Zepto policies: LangGraph intent "
        "router + ChromaDB retrieval + guaranteed Pydantic JSON output."
    ),
)


@app.get("/")
def root() -> dict:
    return {
        "service": "zepto-support-assistant",
        "MOCK_LLM": os.environ.get("MOCK_LLM", "1"),
        "mode": "mock (graded baseline)" if MOCK_LLM else "real-LLM (optional extension)",
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest) -> AnswerResponse:
    """Run one query through the graph and return the validated schema."""
    state = handle_query(request.query)
    return AnswerResponse(
        answer=state["answer"],
        sources=state["sources"],
        confidence=state["confidence"],
    )