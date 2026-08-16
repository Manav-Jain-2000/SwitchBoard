# SwitchBoard Architecture

## System Overview

SwitchBoard is a production-grade multi-LLM orchestration system that intelligently routes queries to the optimal model from a pool of 7 LLMs. The core principle: **not every query needs an expensive model**.

```
                          +-----------------+
                          |   User Query    |
                          +--------+--------+
                                   |
                          +--------v--------+
                          |   Classifier    |  Cerebras Llama 3.1 8B (~200ms)
                          |  (Guardrails)   |  - Greeting detection (regex)
                          +---+----+----+---+  - Critical keyword filter
                              |    |    |      - Ambiguity detection
              can_self_answer |    |    | is_ambiguous
                     +--------+    |    +--------+
                     |             |             |
                     v             |             v
                   [END]           |         +-------+
               (greeting)         |         | HITL  |  interrupt() / resume
                                   |         +---+---+
                                   |             |
                          +--------v-------------v---+
                          |      KNN Router          |  Fireworks nomic-embed (~10ms)
                          |  embed -> cosine sim     |  105 prototypes, top-5 vote
                          |  -> majority vote        |
                          +------+-------------+-----+
                                 |             |
                        single   |             | subtasks[]
                                 |             |
                          +------v------+  +---v-----------+
                          |   Worker    |  | Parallel      |  asyncio.gather()
                          | (1 model)   |  | Workers       |  1 model per subtask
                          +------+------+  +---+-----------+
                                 |             |
                                 |    +--------v--------+
                                 |    |   Aggregator    |  Gemini Flash
                                 |    | (merge outputs) |
                                 |    +--------+--------+
                                 |             |
                          +------v-------------v-----+
                          |     is_critical?         |
                          +------+-------------+-----+
                                 |             |
                            NO   |             | YES
                                 |             |
                          +------v------+  +---v-----------+
                          | set_final   |  |    Judge      |  Gemini Flash
                          |   -> END    |  | (score 0-10)  |  Context-aware strictness
                          +-------------+  +---+-----------+
                                               |
                                    score >= threshold?
                                         |          |
                                        YES         NO
                                         |          |
                                  +------v---+  +---v-----------+
                                  |set_final |  | Escalation    |  Category-aware model
                                  |  -> END  |  | Worker        |  GPT-4o (code) / Opus (medical)
                                  +----------+  +---+-----------+
                                                    |
                                             +------v------+
                                             |  set_final  |
                                             |   -> END    |
                                             +-------------+
```

## Pipeline Stages

### 1. Classifier (`src/agents/classifier.py`)
- **Model:** Cerebras Llama 3.1 8B (cheapest, fastest)
- **Purpose:** Classify query intent, detect greetings, flag ambiguity, identify critical queries
- **Guardrails:**
  - `_is_greeting_or_smalltalk()`: Regex-based exact match prevents complex queries from being self-answered
  - `_is_likely_critical()`: Two-tier keyword system prevents false critical flagging on code queries
    - **Strong keywords** (always flag): drug, stroke, malpractice, surgery, etc.
    - **Weak keywords** (context-dependent): audit, compliance, safety — suppressed when code context words present
- **Output:** `can_self_answer`, `is_critical`, `is_ambiguous`, `subtasks[]`

### 2. HITL Node (`src/agents/hitl.py`)
- **No LLM call** — pure I/O using LangGraph `interrupt()`
- **Purpose:** Pause graph execution for user clarification when query is ambiguous
- **Resume:** Via API `POST /resume` with `Command(resume=answer)`
- **Enrichment:** Appends `[Clarified: user_answer]` to query

### 3. KNN Router (`src/agents/knn_router.py`)
- **Model:** Fireworks nomic-embed-text-v1.5 (embedding only, ~$0.00001/query)
- **Purpose:** Route query to optimal model using semantic similarity
- **Process:**
  1. Embed query via Fireworks (with LRU cache, 1000 entries)
  2. Cosine similarity against 105 prototype vectors (15 per model)
  3. Top-5 nearest neighbors → majority vote
- **Why KNN over LLM routing:** 10x faster, 10x cheaper, deterministic, no hallucination

### 4. Worker / Parallel Workers (`src/agents/worker.py`)
- **Model:** Selected by KNN router (varies per query)
- **Single worker:** Direct LLM call with 30s timeout
- **Parallel workers:** `asyncio.gather()` for subtasks — latency = max (not sum)
- **Cost tracking:** Per-call cost via LiteLLM + manual fallback

### 5. Aggregator (`src/agents/aggregator.py`)
- **Model:** Gemini 2.5 Flash
- **Purpose:** Merge parallel worker outputs into single cohesive response
- **Fallback:** On LLM failure, returns raw concatenated outputs

### 6. Judge (`src/agents/judge.py`)
- **Model:** Gemini 2.5 Flash
- **Purpose:** Quality gate for critical queries (medical/legal/financial)
- **Tiered thresholds:**
  - `critical` (medical/legal): 7.0 — strict, accuracy is paramount
  - `critical_code`: 6.0 — more lenient to avoid costly escalations
- **Category inference:** `_infer_category()` detects code vs medical/legal context
- **Output:** Score 0-10, dimensions (accuracy, completeness, reasoning_depth), escalation instructions

### 7. Escalation Worker (`src/agents/judge.py`)
- **Model:** Category-aware selection
  - `critical` (medical/legal) → Claude Opus ($15/$75 per 1M tokens)
  - `critical_code` → GPT-4o ($2.50/$10 per 1M tokens) — 10x cheaper
  - `default` → GPT-4o
- **Limit:** MAX_ESCALATIONS = 1 (prevents infinite retry loops)

## Model Pool

| Model | Provider | Cost/1M tokens | Role |
|-------|----------|---------------|------|
| Llama 3.1 8B | Cerebras | $0.10 in / $0.10 out | Classifier |
| Llama 3.1 8B | Groq | $0.05 / $0.08 | Simple Q&A |
| Kimi K2 | Groq | $0.20 / $0.20 | Code |
| GPT-OSS 120B | Cerebras | $0.50 / $0.50 | General knowledge |
| Qwen 3 235B | Cerebras | $0.40 / $0.40 | Research |
| GPT-4o | OpenAI | $2.50 / $10.00 | Critical Q&A + code escalation |
| Gemini 2.5 Flash | Google | $0.075 / $0.30 | Math, judge, aggregator |
| Claude Opus | OpenRouter | $15.00 / $75.00 | Critical escalation only |

## Cost Optimization Strategies

1. **Semantic KNN routing** — $0.00001/query (embedding) vs $0.0001 (LLM router)
2. **Model tiering** — 80% of queries → $0.05-$0.50/1M models
3. **Greeting early-exit** — Zero routing cost for greetings
4. **Context-aware critical guardrail** — Prevents code queries from being flagged as critical (avoids unnecessary judge + escalation)
5. **Tiered judge thresholds** — Lower threshold for code (6.0 vs 7.0) reduces escalations
6. **Category-aware escalation** — Code escalates to GPT-4o ($2.50) not Opus ($15.00)
7. **Embedding cache** — LRU cache avoids redundant API calls
8. **Parallel execution** — Subtask latency = max (not sum)

## State Management

Defined in `src/core/state.py` as a TypedDict:

```
SwitchboardState
├── Query: query, enriched_query, original_query
├── Classification: can_self_answer, is_critical, is_ambiguous
├── Clarification: clarifying_question, user_clarification, conversation_turns
├── Routing: subtasks[], selected_models[]
├── Responses: worker_responses[] (Annotated[List, add]), aggregated_response, final_response
├── Judge: judge_score, judge_feedback
├── Escalation: escalation_model, escalation_instruction, escalation_count
├── Metrics: knn_scores, trace[] (Annotated[List, add]), total_cost, total_latency
└── Error: error (Optional)
```

Key design: `worker_responses` and `trace` use LangGraph's `Annotated[List, add]` operator — each node appends to the list rather than replacing it, enabling accumulation across parallel workers.

## Evaluation System

### Benchmark (`src/eval/benchmark.py` + `src/eval/e2e_benchmark.py`)
- 56 queries across 7 categories with expected model routing
- Per-query metrics: routing accuracy, cost, latency, KNN confidence, flow trace
- Per-category metrics: routing accuracy %, avg cost, savings vs baseline, escalation rate, P50 latency
- Advanced metrics: confusion matrix, P50/P95 latency, cost breakdown by stage

### Quality Scorer (`src/eval/quality_scorer.py`)
- Auto-scores responses using Gemini Flash
- Dimensions: accuracy, completeness, relevance, safety
- Integrated into e2e_benchmark for per-query and per-category quality scores

## Directory Structure

```
src/
├── core/
│   ├── state.py          # SwitchboardState TypedDict
│   ├── graph.py          # LangGraph pipeline (9 nodes, conditional edges)
│   ├── config.py         # Models, costs, thresholds, escalation config
│   ├── metrics.py        # Cost calculation (LiteLLM + manual fallback)
│   ├── prototypes.py     # 105 prototype queries (15 per model)
│   └── logging.py        # Structured logging
├── agents/
│   ├── classifier.py     # Classifier + greeting/critical guardrails
│   ├── knn_router.py     # Embed + cosine sim + KNN vote + cache
│   ├── hitl.py           # LangGraph interrupt/resume
│   ├── worker.py         # Single + parallel workers
│   ├── aggregator.py     # Merge parallel outputs
│   └── judge.py          # Tiered judge + category-aware escalation
├── api/
│   └── main.py           # FastAPI + SSE streaming
├── ui/
│   └── app.py            # Streamlit chat UI + trace sidebar
└── eval/
    ├── benchmark.py      # 56 categorized queries
    ├── e2e_benchmark.py  # Full runner with advanced metrics
    └── quality_scorer.py # Auto response quality scoring
```
