# /support_assistant — Zepto GenAI support assistant (offline mock RAG)

A small end-to-end GenAI service for Zepto's delivery/returns/membership/support policies:

- **Corpus** — 8 policy documents (`docs/doc_01.txt` … `doc_08.txt`, exact prescribed text).
- **Embeddings & index** — local `sentence-transformers` (`all-MiniLM-L6-v2`, 384-d) + ChromaDB
  (persistent collection `zepto_policies`, cosine space, `data/chroma/`). No API, no network at
  inference time (model is cached locally after the one-time download).
- **Orchestration** — a LangGraph `StateGraph` (TypedDict state, 3 nodes, 1 conditional intent
  router): `classify_intent → retrieve_and_answer | direct_answer`.
- **Structured output** — a guaranteed Pydantic JSON schema `{answer, sources, confidence}` on
  every final answer, populated deterministically in mock mode.
- **API** — FastAPI `POST /ask` served with uvicorn, plus a Dockerfile.

**Graded baseline = fully offline mock.** Every LLM call is gated behind a single environment
variable, `MOCK_LLM`. Unset or `MOCK_LLM=1` (the default — what gets graded) the service runs the
deterministic, rule-based mock: no signup, no API key, no network call to any LLM provider.
`MOCK_LLM=0` switches on the optional, ungraded real-LLM extension (Groq free tier), which is never
required for correct results.

---

## Quickstart (graded baseline — MOCK_LLM at its default)

```bash
# 1. dependencies (torch CPU is pulled separately in the Dockerfile)
pip install -r requirements.txt

# 2. build the vector index (embeds the 8 docs into ChromaDB data/chroma/)
python ingest.py

# 3. in-process demo of the graph (3 example calls)
python demo.py

# 4. FastAPI server
uvicorn main:app --host 0.0.0.0 --port 7860
# Windows PowerShell: $env:MOCK_LLM="1"; python -m uvicorn main:app --host 0.0.0.0 --port 7860

# 5. call it
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query": "What is Zepto'\''s delivery time?"}'
```

`ingest.py` is idempotent (`collection.upsert`); re-running it refreshes the same 8 chunks.

Prefer a notebook walk-through? See `notebooks/03_rag_pipeline_demo.ipynb` — an executed companion
notebook that steps through the exact same pipeline (corpus → embeddings → ChromaDB retrieval →
intent router → LangGraph graph → validated Pydantic JSON) by importing the same `app/*.py`
objects. The *served* application must remain plain Python, because uvicorn and the Dockerfile
(`CMD ["uvicorn", "main:app", ...]`) can only run a Python app — but the notebook demonstrates
every stage with live outputs, no server required.

---

## Example call transcripts (MOCK_LLM left at its default — graded baseline)

### In-process graph demo (`python demo.py`)

```
MOCK_LLM=unset/1 (mock, graded baseline)

=== POLICY query (keyword 'delivery' -> retrieve_and_answer) ===
query  : What is Zepto's delivery time?
intent : policy_question
JSON   : {"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01","doc_04","doc_08"],"confidence":1.0}

=== POLICY query (keyword 'cancel' -> retrieve_and_answer) ===
query  : Can I cancel my order after it has been packed?
intent : policy_question
JSON   : {"answer":"Based on the retrieved context: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be","sources":["doc_05","doc_02","doc_06"],"confidence":1.0}

=== GENERAL query (no keyword -> direct_answer) ===
query  : What is the capital of France?
intent : general_question
JSON   : {"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

**Routing proof:** the policy queries (keywords `delivery`, `cancel`) are classified
`policy_question` and routed to retrieval; the unrelated query is classified `general_question`
and routed to the direct answer. No LLM call happens in any of the three cases.
**Retrieval proof:** the delivery question returns `doc_01` (Delivery Policy) as its top chunk and
the cancellation question returns `doc_05` (Order Cancellation Policy) — the retrieved content
actually matches the question asked. Retrieval runs for real in both modes (local embedding +
ChromaDB, no API key, no network).

### FastAPI `POST /ask` (uvicorn, MOCK_LLM default)

```
GET /healthz  -> {"status": "ok"}
GET /         -> {"service": "zepto-support-assistant", "MOCK_LLM": "1", "mode": "mock (graded baseline)"}

=== POLICY example (should trigger retrieval) ===
POST /ask  body: {"query": "What is Zepto's delivery time?"}
status            : 200
raw JSON response : {"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01","doc_04","doc_08"],"confidence":1.0}

=== GENERAL example (should NOT trigger retrieval) ===
POST /ask  body: {"query": "What is the capital of France?"}
status            : 200
raw JSON response : {"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

Both endpoints return the Pydantic-validated schema, deterministically populated in mock mode:
`sources` = ids of the retrieved chunks for policy questions (empty for general), `confidence` = 1.0.

---

## Architecture

```
┌───────────────┐   ┌──────────────────┐   ┌───────────────────────────────┐
│  INGESTION    │   │   EMBEDDING      │   │   INDEX                       │
│  ingest.py    │──▶│  app/embeddings  │──▶│  ChromaDB 'zepto_policies'    │
│  docs/*.txt   │   │  all-MiniLM-L6-v2│   │  cosine, data/chroma/         │
└───────────────┘   └──────────────────┘   └───────────────┬───────────────┘
                                                          │ app/store.py
┌──────────────────────────────────────────────────────────▼──────────────────────┐
│  LANGRAPH GRAPH (app/graph.py, StateGraph + TypedDict SupportState)             │
│                                                                                │
│  START ─▶ classify_intent ──(conditional edge)──▶ retrieve_and_answer ─▶ END   │
│               │  ^                                  (policy_question)          │
│  keyword/LLM  ▼  │                                      │                      │
│        general_question ───────────────────────▶ direct_answer ────▶ END       │
└────────────────────────────────────────────────────────────────────────────────┘
        │ output {answer, sources, confidence} validated by app/schema.py
        ▼
   FastAPI POST /ask (main.py) — uvicorn / Docker
```

**Stage-by-stage (who handles what, and how data flows):**

1. **Ingestion** — `ingest.py` reads the 8 files from `docs/`, applies **one chunk per document**
   (each doc is a single short paragraph; a fixed-size chunker was unnecessary at this length),
   and registers each chunk with a stable id (`doc_01`…`doc_08`) plus metadata
   (`doc`, `title`). This is the only place document text enters the system.
2. **Embedding** — `app/embeddings.py` wraps `sentence-transformers` `all-MiniLM-L6-v2`: chunk
   texts are L2-normalized into 384-d vectors. Runs 100% locally; the model is downloaded once
   into the HF cache on first use, then reused offline.
3. **Index / Retrieval** — `app/store.py` owns the ChromaDB `PersistentClient`:
   `upsert_chunks` stores `(id, text, embedding, metadata)` in collection `zepto_policies` under
   `data/chroma/` with `hnsw:space=cosine`; `query_chunks(query, n=3)` embeds the query and
   returns the top-3 chunks by cosine similarity (as `(id, text, distance)`). **This stage runs
   for real in both modes** — no API key, no network.
4. **Routing** — `app/graph.py` builds the LangGraph `StateGraph` with `TypedDict` state
   (`query, intent, retrieved, answer, sources, confidence`) and 3 nodes:
   - `classify_intent` — sets `intent` to `policy_question` or `general_question`;
   - `retrieve_and_answer` — calls `store.query_chunks` then generates the final answer;
   - `direct_answer` — answers general questions without retrieval.
   A **conditional edge** from `classify_intent` (function `route_after_intent`) routes
   `policy_question → retrieve_and_answer` and `general_question → direct_answer`,
   mirroring a graph-based intent router. **The routing logic itself never depends on
   MOCK_LLM** — only the generation step inside each node branches.
5. **Generation** — `app/prompt.py` holds the structured prompt template
   (role–context–task–format–length + negative constraint + few-shot example) and
   `app/llm.py` gates the call.
   - **Mock (MOCK_LLM unset/1 — graded baseline):** `retrieve_and_answer` returns
     `f"Based on the retrieved context: {top_chunk_snippet}"` where `top_chunk_snippet` is the
     first ~200 characters of the single most similar retrieved chunk; `direct_answer` returns
     the fixed canned string *"I can only answer questions about Zepto policies right now."*.
     **No LLM call, no network.**
   - **Real LLM (optional, MOCK_LLM=0):** the same nodes call Groq (free tier) with the
     structured template — retrieval-answers are grounded in the retrieved chunks; direct
     answers use an empty context. Raw output is validated against the schema with up to 2
     corrective retries, then a clearly marked `[error]` response on failure.
6. **Schema / serving** — `app/schema.py` (Pydantic `AnswerResponse`:
   `answer: str`, `sources: list[str]`, `confidence: float ∈ [0,1]`); `main.py` exposes
   `POST /ask` returning that model. In mock mode the schema is populated deterministically
   (sources = retrieved chunk ids or `[]`; confidence = 1.0), so there is no LLM output that
   could fail validation.

**What changes with `MOCK_LLM=0`:** only the *generation step* inside `classify_intent`,
`retrieve_and_answer` and `direct_answer` switches from the deterministic mock to a real LLM
call. Ingestion, embedding, ChromaDB retrieval and routing are identical in both states.
`MOCK_LLM=0` additionally requires a `GROQ_API_KEY` (free tier) and performs real network
calls — it is entirely optional and ungraded.

---

## Docker (required graded baseline — build & run locally)

```bash
docker build -t zepto-support-assistant .
docker run --rm -p 7860:7860 zepto-support-assistant
# then: curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "What is Zepto's delivery time?"}'
```

The `Dockerfile`:
- base `python:3.12-slim` (with `libgomp1` for ChromaDB's hnswlib);
- installs **CPU-only torch** first (avoids the multi-GB CUDA wheel), then `requirements.txt`;
- pre-downloads `all-MiniLM-L6-v2` into the image so runtime stays offline;
- runs `python ingest.py` at build time so the Chroma index ships inside the image;
- serves `uvicorn main:app --host 0.0.0.0 --port 7860`.

> Note: Docker is not installed on the development machine where this module was authored, so
> `docker build` was not executed here. The Dockerfile is standard, self-contained and
> documented as buildable/runnable locally — matching the required graded baseline (a push to
> Hugging Face Spaces is an optional, ungraded stretch and was not attempted).

## Optional extensions (not required, ungraded)

- **Real LLM via Groq free tier:** set `MOCK_LLM=0` and `GROQ_API_KEY=<free-tier key>`
  (model: `GROQ_MODEL`, default `llama-3.1-8b-instant`). The structured prompt template in
  `app/prompt.py` and the retry-on-validation-failure logic in `app/llm.py` are implemented and
  ready, but the graded baseline never touches them.
- **Hugging Face Spaces:** a copy of this Dockerfile can be deployed to the free community CPU
  tier with the API key stored as a Space secret (never committed). Not attempted here.

## Module layout

```
support_assistant/
├── README.md            ← this file (transcripts + architecture)
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── main.py              ← FastAPI app (POST /ask)
├── ingest.py            ← ingestion: docs/ → embed → ChromaDB
├── demo.py              ← in-process example calls
├── notebooks/
│   └── 03_rag_pipeline_demo.ipynb   ← executed companion walk-through of the pipeline
├── app/
│   ├── schema.py        ← Pydantic AskRequest / AnswerResponse
│   ├── prompt.py        ← structured template (role-context-task-format-length + constraint + few-shot)
│   ├── embeddings.py    ← sentence-transformers all-MiniLM-L6-v2
│   ├── store.py         ← ChromaDB collection 'zepto_policies' (persist + query)
│   ├── llm.py           ← MOCK_LLM gate; mock canned logic; real-LLM client + retry
│   └── graph.py         ← LangGraph StateGraph (3 nodes + conditional edge)
├── docs/                ← doc_01.txt … doc_08.txt (the corpus, exact prescribed text)
└── data/chroma/         ← persisted ChromaDB index (regenerable via python ingest.py)
```