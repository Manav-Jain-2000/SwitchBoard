# SwitchBoard — Interview Guide (Explained Simply)

> A beginner-friendly walkthrough of the project, written the way you'd explain it to a classmate who has never seen the code before. Use this to prep for your interview.

---

## 1. What is this project? (The simplest answer)

Imagine you have **8 different AI assistants**. Some are cheap and fast but not very smart (like a calculator). Some are expensive and very smart (like a top doctor). Some are good at code, some are good at math, some are good at medical advice.

**If you ask "What is the capital of Japan?"** — you don't need the $15/million-tokens expensive AI. A $0.05 one can answer this in milliseconds.

**If you ask "Is it safe to mix metformin with ibuprofen?"** — now you want the expensive, careful AI. Mistakes are costly.

**SwitchBoard is the smart traffic cop that decides which AI should answer each question.** It routes 80% of questions to cheap AIs (saving 80-99% on cost) and only sends critical questions to expensive AIs — *and* it double-checks their answers.

---

## 2. The Problem Statement (Why this project exists)

**The pain:**
1. If a company builds a chatbot using **only GPT-4** → it costs a fortune, even for "hi" and "what's 2+2"
2. If they use **only a cheap model** → it's bad at hard things (medical, legal, complex code)
3. Picking the right model *manually* doesn't scale — you'd need a human in the loop
4. Using an **LLM to pick the LLM** is slow (~200ms) and can hallucinate ("I think this needs GPT-4o" — but it doesn't)

**The idea:**
- Use **math (not AI)** to pick the right model — fast, cheap, reliable
- Add **safety nets** so critical queries (medical, legal) get extra quality checks
- Let the system **ask clarifying questions** if a query is vague

That's SwitchBoard in one sentence.

---

## 3. The Big Idea — "Not Every Query Needs a Ferrari"

Benchmark result: on 56 test queries, SwitchBoard achieved **up to 98% cost savings** vs sending everything to GPT-5.

Here's the trick: **80% of real-world questions are easy**. Simple Q&A, basic math, greetings, routine code. Those don't need a $75/million-token model. Only 5-10% of queries — medical advice, legal questions, critical code — truly need the expensive brains.

SwitchBoard sorts them automatically.

---

## 4. How It Works — Step by Step (like a recipe)

Let's say a user types: *"Write a Python function to reverse a linked list."*

### Step 1: Classifier — the bouncer at the door
- Model used: **Cerebras Llama 3.1 8B** (super cheap, super fast — ~200ms)
- Asks: *"Is this a greeting? Is it ambiguous? Is it critical (medical/legal/financial)?"*
- For our example: not a greeting, not ambiguous, not critical → proceed.

### Step 2: Guardrails — safety checks on the classifier
The small classifier sometimes makes mistakes. So we added two guardrails:

**Greeting guardrail (regex):** Even if the classifier says "yes I can answer this myself", we only allow it for actual greetings ("hi", "hello"). This prevents the cheap classifier from accidentally answering hard questions poorly.

**Critical guardrail (keyword check with context awareness):**
- **Strong words** (always flag as critical): *drug, stroke, lawsuit, surgery, malpractice*
- **Weak words** (flag *only* if no code context): *safe, safety, compliance, audit*
- **Code context words**: *write, implement, function, API, deploy*

So *"write a **safe** API endpoint"* is NOT flagged as critical (because of "write" + "API"), but *"is it **safe** to take 2 ibuprofens?"* IS flagged (no code context + strong medical word).

This is a HUGE improvement — without it, any mention of "safe" would trigger expensive judge + escalation.

### Step 3: HITL (Human-in-the-loop) — if the query is vague
If the classifier says the query is ambiguous (e.g., user just says *"help"*), the graph **pauses** using LangGraph's `interrupt()`. The UI shows a clarifying question to the user. When the user answers, we call `/resume` with their answer and the graph continues.

This is a real "pause mid-execution" — not a hack. LangGraph's checkpointing saves the state mid-flow.

### Step 4: KNN Router — the heart of the project ⭐
**This is the most important concept to understand.**

Instead of asking an AI *"which model should handle this?"* (slow, expensive, can hallucinate), we use **math**.

**Setup (done once at startup):**
- We have 7 worker models (Llama, Kimi, GPT-OSS, Qwen, GPT-4o, Gemini Flash, Opus)
- For each model, we wrote **15 example queries it's good at** = 105 total prototype queries
- We send all 105 through an **embedding model** (Fireworks nomic-embed) which turns each into a vector of numbers (think: a "fingerprint" of meaning)
- These 105 vectors are stored in memory — that's our "KNN index"

**At query time:**
1. Embed the incoming query into a vector (~10ms, ~$0.00001)
2. Compute **cosine similarity** between this vector and all 105 prototype vectors
3. Find the **top-5 most similar prototypes**
4. **Majority vote** on which model those 5 belong to → that's the winner

**Why "KNN"?** K-Nearest-Neighbors — a classic machine learning technique. K=5 means we look at the 5 closest matches.

**Why this is genius:**
- ~10ms (vs 200ms if we asked an LLM to route)
- ~$0.00001 (vs $0.0001 if we asked an LLM)
- **Deterministic** — same query always routes the same way
- **Zero hallucination** — it's pure math

For our "reverse a linked list" example, the top-5 matches will all be Kimi K2 prototypes → route to Kimi K2.

**LRU cache bonus:** We cache the last 1000 query embeddings, so if a user sends a repeat query → zero API cost.

### Step 5: Worker — the actual answer
- Single query → one worker makes one LLM call (30s timeout)
- Multi-part query → **parallel workers** using `asyncio.gather()` (all run at the same time)
- Each worker records: the response, the cost, the latency

### Step 6: Aggregator (only if parallel) — merge the results
If we split *"Explain WWI causes AND economic impact"* into 2 subtasks, two models answer in parallel. Then **Gemini Flash** merges them into one clean response — no repetition, all key insights preserved.

### Step 7: Judge — quality gate for critical queries
**This only runs if `is_critical` is true.** For normal queries, skip this and go to final.

- Model: **Gemini 2.5 Flash**
- Scores the response 0-10 on 3 dimensions: *accuracy*, *completeness*, *reasoning_depth*
- **Tiered thresholds**:
  - `critical` (medical/legal) → threshold 7.0 (strict — accuracy matters)
  - `critical_code` → threshold 6.0 (lenient — code is easier to verify)
- **Context-aware strictness**: the judge is told to be harsh on medical, lenient on general

### Step 8: Escalation Worker — safety net
If the judge says score < threshold, we escalate to a **more powerful model**:
- For critical medical/legal → **Claude Opus** ($15/$75 per 1M tokens)
- For critical code → **GPT-4o** ($2.50/$10 — 10x cheaper than Opus)

**Hard limit:** `MAX_ESCALATIONS = 1`. We never escalate twice — prevents infinite loops and runaway costs.

### Step 9: set_final — finalize and return
Cleans up state, sets `final_response`, logs total cost, total latency, and the full trace of every node that ran.

The frontend receives this via **SSE (Server-Sent Events)** — so you see the trace update in real-time as nodes execute.

---

## 5. Tech Stack — What We Used and Why

| Tool | What it does | Why we picked it |
|---|---|---|
| **LangGraph** | Orchestrates the pipeline as a state graph | Conditional edges, parallel execution, `interrupt()` for HITL, `MemorySaver` for checkpointing |
| **LiteLLM** | Unified API for all 8 LLM providers | One `acompletion()` call works for OpenAI, Groq, Cerebras, Gemini, OpenRouter — no provider-specific code |
| **Fireworks AI** | Embedding model (nomic-embed-text-v1.5) | Cheap and fast for KNN routing |
| **scikit-learn** | Cosine similarity math | Industry-standard ML library |
| **numpy** | Vector operations | Standard for numerical work |
| **FastAPI** | Backend web framework | Async-native, SSE streaming, auto docs |
| **Streamlit** | Chat UI + live trace sidebar | Fastest way to build a Python UI |
| **LangSmith** | End-to-end tracing | Shows every node + LLM call in a timeline |
| **pytest + pytest-asyncio** | Testing | 46 tests (7 offline, 39 live-gated on API keys) |
| **Docker + docker-compose** | Packaging | Easy deploy anywhere |

---

## 6. The 8 Models — Know Who Does What

| Model | Cost/1M tokens | Used For |
|---|---|---|
| **Llama 3.1 8B (Cerebras)** | $0.10 | Classifier — speed over brains |
| **Llama 3.1 8B (Groq)** | $0.05 | Simple Q&A (cheapest) |
| **Kimi K2 (Groq)** | $0.20 | Code (low-medium complexity) |
| **GPT-OSS 120B (Cerebras)** | $0.50 | General medium tasks |
| **Qwen 3 235B (Cerebras)** | $0.40 | Research, complex reasoning |
| **Gemini 2.5 Flash** | $0.075 / $0.30 | Math, judge, aggregator |
| **GPT-4o** | $2.50 / $10 | Critical Q&A + code escalation |
| **Claude Opus** | $15 / $75 | Critical escalation ONLY (safety net) |

Notice the price range: Llama-Groq ($0.05) to Opus ($75) = **1500x cost difference**. That's why routing matters.

---

## 7. Problems I Faced and How I Fixed Them

**⚠️ These are the most important stories for the interview. They prove you actually built this.**

### Problem 1: Using an LLM to pick the model was slow and unreliable
**What went wrong:** First version used a small LLM to say *"for this query, use model X"*. That took 200ms, cost $0.0001, and sometimes hallucinated non-existent models.

**How I fixed it:** Switched to **KNN routing with embeddings**. Pre-embed 105 prototype queries (15 per model), cosine-similarity the incoming query against them, top-5 majority vote. 10x faster, 10x cheaper, zero hallucinations.

### Problem 2: Small classifier LLM over-flagged queries as critical
**What went wrong:** The tiny Cerebras 8B model would flag *"write a safe API endpoint"* as a critical medical query because of the word "safe". This triggered unnecessary judge + escalation = wasted money.

**How I fixed it:** Added a **two-tier keyword guardrail** in `classifier.py`:
- Strong words (drug, stroke, surgery) → always critical
- Weak words (safe, compliance, audit) → only critical if there are no code-context words nearby
- If "write" / "API" / "function" appear → suppress the weak critical flag

### Problem 3: Classifier trying to self-answer hard questions
**What went wrong:** The classifier sometimes returned *"can_self_answer = true"* for real questions, and then gave a bad answer using the tiny 8B model.

**How I fixed it:** Added a **regex-based greeting whitelist**. Even if the classifier says "I can answer this", we only allow early-exit if the query exactly matches a greeting pattern ("hi", "hello", "good morning"). Otherwise we force-route to the proper worker.

### Problem 4: Parallel workers overwriting each other's state
**What went wrong:** When two parallel workers finished, their `worker_responses` outputs were overwriting each other — last one wins.

**How I fixed it:** Used LangGraph's **Annotated reducer pattern**: `Annotated[List[Dict], add]`. Each parallel write gets *appended* instead of overwritten. Same fix for the `trace` list.

### Problem 5: Escalation for code questions was way too expensive
**What went wrong:** Originally, every failed judge call escalated to Claude Opus ($75/1M output tokens). But code questions don't need Opus — GPT-4o is plenty good.

**How I fixed it:** **Category-aware escalation**:
- `critical` (medical/legal) → Opus
- `critical_code` → GPT-4o (10x cheaper)

Also added **tiered judge thresholds**: critical_code uses threshold 6.0 (more lenient), so it escalates less often.

### Problem 6: Embedding API calls added up fast
**What went wrong:** Every query triggered an embedding call. For repeat queries, this was wasteful.

**How I fixed it:** Added an **LRU cache** (1000 entries) for query embeddings. Repeat queries are free.

### Problem 7: Aggregator might fail on malformed output
**What went wrong:** If Gemini returned an error during aggregation, the user saw nothing.

**How I fixed it:** On aggregator failure, fall back to **raw concatenated outputs** of the parallel workers. The user gets something useful instead of an empty response.

### Problem 8: Infinite escalation loops
**What went wrong:** If the judge kept rejecting a response, we could escalate forever.

**How I fixed it:** `MAX_ESCALATIONS = 1` constant + a check in `route_from_judge()`. After one escalation, accept whatever we have and return.

### Problem 9: Ambiguous queries got useless answers
**What went wrong:** If the user typed just *"help"*, the system picked a random model and answered badly.

**How I fixed it:** **Human-in-the-loop (HITL)** via LangGraph's `interrupt()`. Pause the graph, show the user a clarifying question, resume with their answer enriched into the query.

---

## 8. Jargon Dictionary (memorize these)

| Term | Plain-English meaning |
|---|---|
| **Orchestration** | Deciding which tool to use and in what order |
| **LLM** | Large Language Model (ChatGPT, Claude, Gemini are all LLMs) |
| **Routing** | Picking which model handles a query |
| **KNN** | K-Nearest-Neighbors — find the K most similar items using math |
| **Embedding** | Turning text into a list of numbers that represent its meaning |
| **Cosine similarity** | A math formula for how close two vectors are (like a similarity score from -1 to 1) |
| **Prototype** | An example query representing "what this model is good at" |
| **Classifier** | A small AI that labels queries (is this critical? ambiguous?) |
| **Guardrail** | A safety check that catches the AI's mistakes |
| **HITL** | Human-in-the-loop — pause for human input |
| **Judge** | An AI that scores another AI's output |
| **Escalation** | Retrying with a more powerful (more expensive) model |
| **Fan-out / Fan-in** | Splitting one task into parallel tasks (fan-out) and merging results (fan-in) |
| **Reducer** | A rule for combining updates to the same state field |
| **Critical query** | Medical, legal, financial, or safety-related (high stakes) |
| **asyncio.gather** | Python way to run multiple async functions in parallel |
| **SSE (Server-Sent Events)** | One-way streaming from server to browser — good for live updates |
| **LRU cache** | Least-Recently-Used cache — drops old entries when full |
| **Trace** | Timeline of every step the system took |
| **LiteLLM** | A library that gives one interface for 100+ LLM providers |
| **State graph** | A diagram of nodes connected by edges, like a flowchart |
| **Conditional edge** | A graph edge that's chosen based on state |
| **Checkpointing** | Saving state so you can pause and resume |
| **Streamlit** | A Python library for quickly building web UIs |
| **FastAPI** | A Python async web framework |

---

## 9. 90-Second Interview Answer ("Walk me through your project")

Practice saying this out loud:

> "SwitchBoard is an intelligent multi-LLM orchestrator. The core idea is that not every question needs a $15-per-million-tokens model — *'What's the capital of Japan?'* can be answered by a $0.05 model just as well. So I built a system that routes queries to the cheapest capable model from a pool of 7 LLMs.
>
> A query first hits a classifier — Cerebras Llama 8B, chosen for speed. It tags the query as greeting, ambiguous, or critical. I added regex guardrails on top because the tiny classifier over-flags things — for example, *'write a safe API'* shouldn't be flagged as a medical-safety query just because of the word 'safe'. So I check if there are code-context words nearby before honoring the critical flag.
>
> If the query is ambiguous, the graph **pauses** using LangGraph's `interrupt()`, asks the user a clarifying question, and resumes on `/resume`. That's human-in-the-loop.
>
> Otherwise, we hit the **KNN router** — the main innovation. Instead of using an LLM to pick the model (slow, hallucinates), I embed the query with Fireworks nomic-embed and compare it to 105 pre-embedded prototype queries (15 per model). Top-5 cosine similarity → majority vote. Takes 10ms, costs a hundredth of a cent, zero hallucinations.
>
> The selected worker calls the LLM via LiteLLM, which gives me one unified API for all 8 providers. If the query has multiple subtasks, I fan out with `asyncio.gather()` and an aggregator (Gemini Flash) merges the responses. The parallel writes to state work because I use LangGraph's `Annotated[List, add]` reducer.
>
> For critical queries only — medical, legal, financial — a judge (Gemini Flash) scores the response 0-10. If it fails the tiered threshold (7 for medical, 6 for code), it escalates to a stronger model — GPT-4o for code, Opus for medical. There's a hard cap of 1 escalation to prevent runaway costs.
>
> End-to-end, I measured up to 98% savings vs sending everything to GPT-5, on a 56-query benchmark. The UI is Streamlit with a real-time trace sidebar so you can see every node as it executes."

---

## 10. Common Interview Questions + Simple Answers

**Q: Why KNN routing instead of an LLM router?**
A: Three reasons — speed (10ms vs 200ms), cost ($0.00001 vs $0.0001), and **determinism**. KNN is pure math, so the same query always routes the same way. LLMs can hallucinate non-existent models or waver between calls. For a production routing layer, predictability matters.

**Q: Why 15 prototypes per model and not more?**
A: More prototypes = better accuracy but more embedding cost at startup and more comparisons per query. I tested 10 vs 15 vs 20 — 15 was the sweet spot where routing accuracy plateaued. Could tune further with real usage data.

**Q: What happens if all the prototypes for a model aren't in the top-5?**
A: Then that model won't win the vote — which is correct behavior. If no prototype of model X is near the query, model X is probably not the right fit.

**Q: How do you handle model failures?**
A: Three layers: (1) 30-second timeout on every worker call, (2) try/except around every LLM call with graceful fallback messages, (3) aggregator falls back to raw concatenated outputs if the merge step fails.

**Q: How do you prevent runaway costs?**
A: `MAX_ESCALATIONS = 1`. Category-aware escalation (code → GPT-4o, not Opus). Tiered judge thresholds so code escalates less often. Critical guardrail that prevents over-flagging. Embedding cache.

**Q: Why LangGraph instead of a simple if/else pipeline?**
A: Three features I needed: (1) `interrupt()` for HITL — pause mid-execution, (2) `MemorySaver` checkpointing so HITL can resume after the user answers, (3) `Annotated[List, add]` reducers for parallel worker state safety. Building those from scratch would be a full side project.

**Q: Why LiteLLM over calling each provider's SDK directly?**
A: One unified `acompletion()` interface for OpenAI, Cerebras, Groq, Google, OpenRouter. Auto cost tracking. If I used 8 different SDKs, switching models would mean 8 different code paths. LiteLLM = one code path.

**Q: How is the judge context-aware?**
A: The judge's prompt includes a strictness mode — STRICT for critical (medical/legal) or LENIENT for standard. Plus the threshold itself changes per category: 7.0 for medical, 6.0 for code. So the same response that would fail as a medical answer might pass as a code answer.

**Q: What's the weakest part of the system?**
A: Two things. (1) The prototype queries are hand-written — they could be learned from real usage data with labeled outcomes. (2) The LRU cache is per-process, so if I scaled to multiple API servers, each would have its own cache. Redis would fix that.

**Q: How would you scale this to millions of queries/day?**
A: Three changes: (1) Move embedding cache from Python LRU to Redis, (2) Pre-warm the KNN index in every worker pod at startup, (3) Use a proper vector database (Pinecone, Qdrant) for the prototype search if prototypes grow beyond a few thousand. KNN on 105 vectors is trivial; on 100k it's not.

**Q: What would you build next?**
A: A feedback loop. Right now I hand-pick prototypes. I'd log every query's routing decision, the judge's score, and user thumbs up/down, then retrain the prototype set based on which queries actually succeeded. Self-improving routing.

---

## 11. Quick Confidence Tips for the Interview

1. **Lead with the core insight.** Your hook is: *"Not every query needs an expensive model — I saved 80-99% on costs by routing smartly."* Say that first.
2. **Explain KNN routing in plain English.** Analogy: *"Imagine each model has 15 sample questions it's good at. When a new question comes in, I find the 15 most similar sample questions and vote on which model to use."*
3. **Own the problems.** The guardrails you added (critical keyword filter, greeting regex) show you **ran the system**, saw bugs, and fixed them. That's what interviewers love.
4. **Don't memorize — understand.** If you understand *why* each piece exists, you can handle any follow-up.
5. **Be honest about weaknesses.** Saying *"the prototype queries are hand-written; I'd learn them from data next"* sounds mature.

---

## 12. The One-Liner Summary

> **"SwitchBoard is a multi-LLM orchestrator that uses KNN routing over embeddings — not another LLM — to pick the cheapest capable model from a pool of 7. It adds a quality judge for critical queries and escalates cost-consciously, saving up to 98% vs always using GPT-5. Built with LangGraph, LiteLLM, Fireworks embeddings, FastAPI, and Streamlit."**

That's it. Good luck! 🚀
