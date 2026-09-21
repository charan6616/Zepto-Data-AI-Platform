"""LLM gate.

Every LLM call in this module is gated behind a single environment variable:

* MOCK_LLM unset or MOCK_LLM=1  -> deterministic, rule-based mock (graded baseline).
  No signup, no API key, no network call to any LLM provider.
* MOCK_LLM=0                    -> optional extension: call a real LLM
  (Groq free tier, or any genuinely free OpenAI-compatible endpoint).

The mock branch is what gets graded; the real-LLM branch must never be required.
"""

import json
import os
import re

MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"

# Fixed canned strings used by the graded mock baseline.
DIRECT_ANSWER_CANNED = "I can only answer questions about Zepto policies right now."
POLICY_ANSWER_TEMPLATE = "Based on the retrieved context: {snippet}"

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def real_llm_chat(messages: list[dict], temperature: float = 0.0) -> str:
    """Optional MOCK_LLM=0 extension: chat completion via an OpenAI-compatible API.

    Raises if no API key is configured, so the mock baseline can never
    accidentally depend on this path.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 requires GROQ_API_KEY (free tier). Set MOCK_LLM back to 1 "
            "to use the graded offline mock."
        )
    import httpx

    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _extract_json(raw: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in LLM output")
    return json.loads(raw[start : end + 1])


def generate_with_validation(prompt: str, retries: int = 2) -> dict:
    """Optional MOCK_LLM=0 path: generate + validate against the JSON schema.

    If the raw LLM output fails to validate, retry up to *retries* (2) additional
    times with a corrective instruction, then give up with a clearly marked error
    response. Never used by the graded mock baseline.
    """
    messages = [{"role": "user", "content": prompt}]
    last_error = None
    for attempt in range(retries + 1):
        try:
            raw = real_llm_chat(messages)
            data = _extract_json(raw)
            return {
                "answer": str(data["answer"]),
                "sources": [str(s) for s in data.get("sources", [])],
                "confidence": float(data["confidence"]),
            }
        except Exception as err:  # validation or transport failure -> corrective retry
            last_error = err
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous output failed schema validation: {err}. "
                        'Return ONLY a JSON object with keys "answer" (string), '
                        '"sources" (array of strings), "confidence" (float 0-1). '
                        "No prose, no markdown fences."
                    ),
                }
            )
    return {
        "answer": f"[error] LLM output failed schema validation after {retries + 1} attempts "
                  f"({last_error}).",
        "sources": [],
        "confidence": 0.0,
    }