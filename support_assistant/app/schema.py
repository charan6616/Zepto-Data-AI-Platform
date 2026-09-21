"""Pydantic models shared by the LangGraph flow and the FastAPI layer."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    query: str = Field(..., description="The customer's question to the assistant.")


class AnswerResponse(BaseModel):
    """Guaranteed JSON output schema for every answer.

    Populated deterministically in mock mode (no LLM output exists to validate);
    for the optional MOCK_LLM=0 extension, LLM output is validated against this
    schema with up to 2 corrective retries before an error response is returned.
    """

    answer: str = Field(..., description="The final answer text.")
    sources: list[str] = Field(
        default_factory=list,
        description="Chunk/document IDs used to ground the answer "
        "(empty for general_question answers).",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the answer, between 0 and 1."
    )