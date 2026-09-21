"""Demo: run example queries through the graph and print the raw JSON responses.

Usage:  python demo.py     (MOCK_LLM at its default -> graded mock baseline)

Queries are chosen so the keyword heuristic (exactly the mandated keyword list:
"delivery", "return", "refund", "membership", "tracking", "cancel", "gift card",
"support hours") demonstrably routes one query to retrieval and one to the direct
answer, with no LLM call either way.
"""

from app.graph import handle_query
from app.llm import MOCK_LLM
from app.schema import AnswerResponse

DEMO_QUERIES = [
    (
        "POLICY query (keyword 'delivery' -> retrieve_and_answer)",
        "What is Zepto's delivery time?",
    ),
    (
        "POLICY query (keyword 'cancel' -> retrieve_and_answer)",
        "Can I cancel my order after it has been packed?",
    ),
    (
        "GENERAL query (no keyword -> direct_answer)",
        "What is the capital of France?",
    ),
]


def main() -> None:
    print(f"MOCK_LLM={'unset/1 (mock, graded baseline)' if MOCK_LLM else '0 (real LLM)'}\n")
    for tag, query in DEMO_QUERIES:
        state = handle_query(query)
        response = AnswerResponse(
            answer=state["answer"],
            sources=state["sources"],
            confidence=state["confidence"],
        )
        print(f"=== {tag} ===")
        print(f"query  : {query}")
        print(f"intent : {state['intent']}")
        print("JSON   : " + response.model_dump_json())
        print()


if __name__ == "__main__":
    main()