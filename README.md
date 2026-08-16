# SwitchBoard

**A semantic switchboard for LLM traffic.** Every incoming query is embedded, matched against a labelled prototype index, and patched through to the cheapest model in the pool that can actually answer it — with a clarification loop for vague requests and a quality judge guarding the answers that matter.

Built on LangGraph + LiteLLM. 8 models, 6 providers, one routing decision that costs ~10ms and a fraction of a cent.

---

## The problem

A chatbot backed by a single frontier model pays frontier prices for *everything*. "hi", "what's 2+2", and "is it safe to take ibuprofen with metformin" all bill at the same rate — but only one of them needs a frontier model.

The obvious fix is to let an LLM pick the model. That trades one problem for another: you add ~200ms and another API bill to every request, and the router itself can hallucinate its choice.

**SwitchBoard routes with math instead.** Query goes in, embedding comes out, cosine similarity against 105 labelled prototypes decides where it goes. No LLM in the routing path. Deterministic, ~10ms, ~$0.00001.

---

## How it works

```
                            User query
                                │
                     ┌──────────▼──────────┐
                     │     Classifier      │  Cerebras Llama 3.1 8B (~200ms)
                     │  + regex guardrails │  greeting? ambiguous? critical?
                     └──┬───────┬───────┬──┘
        can_self_answer │       │       │ is_ambiguous
                        ▼       │       ▼
                      [END]     │    ┌──────┐  LangGraph interrupt()
                   (greeting)   │    │ HITL │  → user clarifies → resume
                                │    └───┬──┘
                     ┌──────────▼────────▼─────┐
                     │       KNN Router        │  Fireworks nomic-embed (~10ms)
                     │  embed → cosine sim →   │  105 prototypes, top-5 vote
                     │  majority vote          │
                     └────┬───────────────┬────┘
                 single   │               │  subtasks[]
                          ▼               ▼
                    ┌──────────┐   ┌──────────────┐   asyncio.gather()
                    │  Worker  │   │  Parallel    │   1 model per subtask
                    │ (1 model)│   │  Workers     │
                    └─────┬────┘   └──────┬───────┘
                          │               ▼
                          │        ┌─────────────┐  Gemini Flash
                          │        │ Aggregator  │  merge subtask answers
                          │        └──────┬──────┘
                     ┌────▼──────────────▼────┐
                     │     is_critical?       │
                     └────┬──────────────┬────┘
                       NO │              │ YES
                          ▼              ▼
                     ┌─────────┐   ┌───────────┐  Gemini Flash, score 0–10
                     │  return │   │   Judge   │  strict for medical/legal
                     └─────────┘   └─────┬─────┘  lenient for code
                                score < threshold?
                                   ┌─────┴─────┐
                                  NO           YES
                                   ▼            ▼
                              ┌─────────┐  ┌──────────────┐
                              │  return │  │  Escalation  │  GPT-4o (code)
                              └─────────┘  │    Worker    │  Opus  (medical/legal)
                                           └──────────────┘  max 1 hop
```

### The five decisions

**1. Classify before routing.** A tiny Cerebras model plus regex guardrails answers three questions: can I just answer this myself (greetings, small talk), is this too vague to route, and is this critical? Greetings exit immediately without touching the pool at all.

**2. Route with KNN, not an LLM.** `src/core/prototypes.py` holds 15 hand-written example queries per model — 105 total, embedded once at startup into an in-memory index. Each incoming query is embedded (LRU-cached, 1000 entries), scored by cosine similarity against every prototype, and the top-5 neighbours cast a majority vote. Pure `scikit-learn`, no inference call, no hallucination surface.

**3. Ask when unsure.** Ambiguous queries hit a LangGraph `interrupt()`. The graph suspends mid-execution, the API streams a clarifying question to the client, and `Command(resume=...)` picks the run back up with the enriched query. State survives via `MemorySaver` checkpointing.

**4. Fan out when the query has parts.** If the classifier decomposes a query into subtasks, each subtask is routed independently and dispatched concurrently through `asyncio.gather()`, then merged by a Gemini Flash aggregator. A single query can end up spanning three different providers.

**5. Judge only what's worth judging.** Running a judge on every response doubles your cost for no benefit. SwitchBoard only invokes it when a keyword guardrail flags the query as genuinely critical — and that guardrail is context-aware: *"is this JWT implementation safe?"* is a code question, not a safety question, and shouldn't burn Opus tokens. Judge strictness and escalation target both vary by category.

---

## The model pool

| Model | Provider | $/1M in → out | Role |
|---|---|---|---|
| Llama 3.1 8B | Cerebras | $0.10 | Classifier — latency over intelligence |
| Llama 3.1 8B | Groq | $0.05 → $0.08 | Simple Q&A |
| Llama 4 Scout 17B | Groq | $0.20 | Code, low–medium complexity |
| GPT-OSS 120B | Groq | $0.50 | General medium-difficulty tasks |
| Qwen 3 235B | Cerebras | $0.40 | Research, complex reasoning |
| Gemini 2.5 Flash | Google | $0.075 → $0.30 | Math, judge, aggregator |
| GPT-4o | OpenRouter | $2.50 → $10.00 | Critical Q&A, default escalation |
| Claude Opus 4.6 | OpenRouter | $15 → $75 | Medical/legal escalation only |
| nomic-embed-text-v1.5 | Fireworks | — | Routing embeddings |

The spread is the point: the cheapest worker is **300×** cheaper per input token than the most expensive one. Routing correctly is worth more than any single model choice.

---

## Cost behaviour

`src/eval/` ships a 56-query benchmark across 7 categories with expected routing labels, measuring per-category routing accuracy, real cost (via LiteLLM token accounting with a manual fallback in `core/metrics.py`), and savings against a configurable frontier baseline.

| Category | Queries | Routing accuracy | Avg cost | Baseline | Savings |
|---|---|---|---|---|---|
| Simple Q&A | 9 | ~90% | ~$0.00003 | $0.002 | **~98%** |
| Math | 7 | ~86% | ~$0.0001 | $0.004 | **~97%** |
| General | 9 | ~78% | ~$0.001 | $0.006 | **~83%** |
| Research | 8 | ~75% | ~$0.003 | $0.015 | **~80%** |
| Code | 9 | ~80% | ~$0.002 | $0.008 | **~75%** |
| Critical code | 6 | ~67% | ~$0.008 | $0.020 | **~60%** |
| Critical | 8 | ~88% | ~$0.005 | $0.012 | **~58%** |

> Indicative numbers from the original benchmark run. Provider pricing and model availability move constantly — run `cd src && python -m eval.e2e_benchmark` against your own keys before quoting anything.

Note the shape rather than the digits: savings are largest exactly where volume is largest (simple Q&A, math) and smallest where correctness matters most (critical). That is the intended trade, not an accident.

---

## Stack

| Component | Why |
|---|---|
| **LangGraph** | Conditional edges, parallel fan-out, and `interrupt()`/`resume` for human-in-the-loop — the piece that makes the clarification loop tractable |
| **LiteLLM** | One `acompletion()` signature across 6 providers, plus automatic token and cost accounting |
| **Fireworks AI** | `nomic-embed-text-v1.5` for routing embeddings |
| **scikit-learn** | Cosine similarity for the KNN vote |
| **FastAPI** | Async backend, SSE streaming of live node traces |
| **Streamlit** | Chat UI with a real-time agent trace sidebar and KNN score bar chart |
| **LangSmith** | Optional end-to-end tracing of every node and LLM call |

---

## Quickstart

```bash
git clone https://github.com/Manav-Jain-2000/SwitchBoard.git
cd SwitchBoard

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # add your provider keys
python main.py
```

API on `http://localhost:8000` (Swagger at `/docs`), UI on `http://localhost:8501`.

Or with Docker:

```bash
docker compose up --build
```

### Keys you'll need

`FIREWORKS_API_KEY` (embeddings — required, nothing routes without it), `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`. `LANGSMITH_API_KEY` is optional. See `.env.example`.

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send a query; returns an SSE stream of node traces followed by the final answer |
| `POST` | `/resume` | Resume a HITL-interrupted run with the user's clarification |
| `GET` | `/trace/{session_id}` | Full trace for a session |
| `GET` | `/models` | Model pool with costs and prototype counts |
| `GET` | `/health` | Health check plus KNN index status |

Every node emits a trace entry (`node`, `action`, `detail`, `timestamp`) as it executes, so the UI shows routing decisions live rather than after the fact.

---

## Testing

```bash
pip install pytest pytest-asyncio pytest-cov
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

**46 tests collected.** 7 run offline with no configuration; the other 39 are marked `@pytest.mark.live` and auto-skip unless every provider key is present in `.env` (`conftest.py` handles the gate). A bare `pytest tests/ -q` on a fresh clone therefore reports `7 passed, 39 skipped` — that is the expected result, not a failure. Fill in `.env` to exercise the full suite.

Coverage spans: classifier guardrails (greeting detection, ambiguity, context-aware critical filtering), KNN routing and index construction, embedding cache behaviour, worker model selection and timeout/error paths, parallel dispatch and cost accumulation, judge thresholds and category-aware escalation, aggregator merging and failure fallback, HITL interrupt/resume, graph edge routing, API endpoints, and benchmark data integrity.

---

## Benchmarking

```bash
cd src
python -m eval.e2e_benchmark                       # all 56 queries
python -m eval.e2e_benchmark --limit 14            # quick pass
python -m eval.e2e_benchmark --limit 21 --query-timeout-s 60
```

---

## Layout

```
SwitchBoard/
├── src/
│   ├── core/
│   │   ├── state.py        # SwitchboardState — the TypedDict every node reads and writes
│   │   ├── graph.py        # LangGraph assembly: nodes, conditional edges, checkpointing
│   │   ├── config.py       # Model IDs, cost tables, per-category thresholds
│   │   ├── prototypes.py   # 105 labelled prototype queries (15 per model)
│   │   ├── metrics.py      # Cost accounting, LiteLLM with manual fallback
│   │   └── logging.py      # Structured logging
│   ├── agents/
│   │   ├── classifier.py   # Cerebras classifier + greeting/critical guardrails
│   │   ├── knn_router.py   # Embedding cache, index build, top-5 cosine vote
│   │   ├── hitl.py         # interrupt() / Command(resume=) clarification node
│   │   ├── worker.py       # Single worker + parallel fan-out
│   │   ├── aggregator.py   # Merge parallel worker outputs
│   │   └── judge.py        # Context-aware scoring + escalation worker
│   ├── api/main.py         # FastAPI, SSE streaming, lifespan index build
│   ├── ui/app.py           # Streamlit chat + live trace sidebar
│   └── eval/               # 56-query benchmark + E2E runner + quality scorer
├── tests/                  # 46 pytest tests (7 offline, 39 live-gated)
├── main.py                 # Runs backend and UI together
├── Dockerfile / docker-compose.yml
└── ARCHITECTURE.md         # Deeper design notes
```

---

## License

MIT — see [LICENSE](LICENSE).
