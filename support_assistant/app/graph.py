"""LangGraph orchestration: classify -> route -> (retrieve & answer | direct answer).

* State: TypedDict shared across nodes.
* 3 nodes: classify_intent, retrieve_and_answer, direct_answer.
* 1 conditional edge: classify_intent -> retrieve_and_answer (policy_question)
  or -> direct_answer (general_question). Routing itself never depends on
  MOCK_LLM — only the generation step inside each node branches on it:
  mock branch = graded baseline (deterministic, no LLM), real-LLM branch =
  optional MOCK_LLM=0 extension.

Retrieval (query embedding + ChromaDB cosine search) always runs for real in
both modes, because embeddings/ChromaDB need no API key and no network call.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import store
from .llm import (
    DIRECT_ANSWER_CANNED,
    MOCK_LLM,
    POLICY_ANSWER_TEMPLATE,
    generate_with_validation,
)
from .prompt import build_prompt

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


class SupportState(TypedDict):
    query: str
    intent: str                       # "policy_question" | "general_question"
    retrieved: list[tuple[str, str, float]]  # (chunk_id, text, cosine distance)
    answer: str
    sources: list[str]
    confidence: float


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def _mock_classify(query: str) -> str:
    """Graded baseline: keyword heuristic, NO LLM call."""
    q = query.lower()
    return "policy_question" if any(k in q for k in POLICY_KEYWORDS) else "general_question"


def classify_intent(state: SupportState) -> dict:
    query = state["query"]
    if MOCK_LLM:
        return {"intent": _mock_classify(query)}

    # Optional MOCK_LLM=0 extension: ask the LLM, falling back to the heuristic.
    try:
        resp = generate_with_validation(
            build_prompt(
                context="(no retrieval in this step)",
                question=f'Classify this query as one of: "policy_question" '
                         f'or "general_question". Query: {query}',
            )
        )
        return {"intent": resp["answer"].strip().lower()}
    except Exception:
        return {"intent": _mock_classify(query)}


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer (policy_question)
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: SupportState) -> dict:
    query = state["query"]
    # Retrieval ALWAYS runs for real (both modes): local embedding + ChromaDB.
    results = store.query_chunks(query, n=3)
    top_id, top_text, _ = results[0]
    top_snippet = top_text[:200]

    if MOCK_LLM:
        # Graded baseline: canned templated answer over the top retrieved chunk.
        return {
            "retrieved": results,
            "answer": POLICY_ANSWER_TEMPLATE.format(snippet=top_snippet),
            "sources": [r[0] for r in results],
            "confidence": 1.0,
        }

    # Optional MOCK_LLM=0 extension: ground the real LLM on the retrieved chunks.
    context = "\n\n".join(f"[{cid}] {txt}" for cid, txt, _ in results)
    resp = generate_with_validation(build_prompt(context=context, question=query))
    return {
        "retrieved": results,
        "answer": resp["answer"],
        "sources": resp["sources"] or [r[0] for r in results],
        "confidence": resp["confidence"],
    }


# ---------------------------------------------------------------------------
# Node 3: direct_answer (general_question)
# ---------------------------------------------------------------------------
def direct_answer(state: SupportState) -> dict:
    if MOCK_LLM:
        # Graded baseline: fixed canned string, no LLM call.
        return {"answer": DIRECT_ANSWER_CANNED, "sources": [], "confidence": 1.0}

    # Optional MOCK_LLM=0 extension: LLM direct answer, no retrieval.
    resp = generate_with_validation(
        build_prompt(
            context="(no retrieval: this is a general question)",
            question=state["query"],
        )
    )
    return {"answer": resp["answer"], "sources": [], "confidence": resp["confidence"]}


# ---------------------------------------------------------------------------
# Conditional edge (intent router) — never depends on MOCK_LLM
# ---------------------------------------------------------------------------
def route_after_intent(state: SupportState) -> str:
    return (
        "direct_answer"
        if state["intent"] == "general_question"
        else "retrieve_and_answer"
    )


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(SupportState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_node("direct_answer", direct_answer)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    builder.add_edge("retrieve_and_answer", END)
    builder.add_edge("direct_answer", END)
    return builder.compile()


graph = build_graph()


def handle_query(query: str) -> dict:
    """Run the query through the graph and return answer/sources/confidence."""
    state = graph.invoke({"query": query})
    return {
        "answer": state["answer"],
        "sources": state.get("sources", []),
        "confidence": state.get("confidence", 1.0),
        "intent": state.get("intent", ""),
    }