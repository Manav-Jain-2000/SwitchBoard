"""End-to-end test suite for SwitchBoard.

Single file, exercises every model and every feature with clearly labeled sections.

Run subsets:
    pytest tests/test_e2e.py -v -k provider     # just provider key checks
    pytest tests/test_e2e.py -v -k models       # just per-model invocations
    pytest tests/test_e2e.py -v -k graph        # just the LangGraph paths
    pytest tests/test_e2e.py -v -k api          # just the FastAPI endpoints

Network calls cost trivial $$ (< $0.01 for the whole suite).
Tests marked @pytest.mark.live auto-skip if .env keys are missing (handled in conftest).
"""

import asyncio
import json
import os

import httpx
import litellm
import pytest

from agents import knn_router as knn_mod
from agents.aggregator import aggregator_node
from agents.classifier import _is_greeting_or_smalltalk, _is_likely_critical, classifier_node
from agents.judge import judge_node
from agents.knn_router import build_knn_index, semantic_route
from agents.worker import parallel_worker_node, worker_node
from api.main import app
from core.config import (
    AGGREGATOR_MODEL,
    JUDGE_MODEL,
    MODEL_CLASSIFIER,
    MODEL_EMBED,
    MODEL_GEMINI_FLASH,
    MODEL_GPT4O,
    MODEL_GPT_OSS,
    MODEL_KIMI_K2,
    MODEL_LLAMA_GROQ,
    MODEL_OPUS,
    MODEL_QWEN_235B,
)
from core.graph import switchboard_graph

litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# Section 0 — Pure-Python guards (no network, always run)
# ---------------------------------------------------------------------------

class TestPureLogic:
    def test_greeting_detection_positive(self):
        for q in ("hi", "hello", "hey there", "good morning", "what's up"):
            assert _is_greeting_or_smalltalk(q), f"expected greeting: {q!r}"

    def test_greeting_detection_negative(self):
        for q in ("what is python", "write a function", "explain quantum mechanics"):
            assert not _is_greeting_or_smalltalk(q), f"not a greeting: {q!r}"

    def test_critical_detection_strong_words(self):
        assert _is_likely_critical("Is it safe to take ibuprofen with blood thinners?")
        assert _is_likely_critical("What are the symptoms of a stroke?")

    def test_critical_detection_skips_code_context(self):
        # 'safe' is a weak critical word; in a code/security context it should NOT trigger
        assert not _is_likely_critical("Write a security audit checklist for a REST API")
        assert not _is_likely_critical("Explain JWT authentication for an Express app")

    def test_critical_detection_negative(self):
        for q in ("what is the capital of japan", "explain TCP/IP", "write quicksort in python"):
            assert not _is_likely_critical(q)


# ---------------------------------------------------------------------------
# Section 1 — Provider key health (raw HTTP, no litellm wrapping)
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestProviderKeys:
    async def test_fireworks_key(self):
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                "https://api.fireworks.ai/inference/v1/models",
                headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"},
            )
        assert r.status_code == 200, r.text

    async def test_groq_key(self):
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            )
        assert r.status_code == 200, r.text

    async def test_cerebras_key(self):
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                "https://api.cerebras.ai/v1/models",
                headers={"Authorization": f"Bearer {os.environ['CEREBRAS_API_KEY']}"},
            )
        assert r.status_code == 200, r.text

    async def test_gemini_key(self):
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={os.environ['GEMINI_API_KEY']}"
            )
        assert r.status_code == 200, r.text

    async def test_openrouter_key(self):
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            )
        assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Section 2 — Each LLM invocable through LiteLLM
# ---------------------------------------------------------------------------

ALL_CHAT_MODELS = [
    pytest.param(MODEL_CLASSIFIER, id="classifier:cerebras-llama3.1-8b"),
    pytest.param(MODEL_LLAMA_GROQ, id="worker:groq-llama-3.1-8b"),
    pytest.param(MODEL_KIMI_K2, id="worker:groq-llama-4-scout"),
    pytest.param(MODEL_GPT_OSS, id="worker:groq-gpt-oss-120b"),
    pytest.param(MODEL_QWEN_235B, id="worker:cerebras-qwen-235b"),
    pytest.param(MODEL_GPT4O, id="worker:openrouter-gpt-4o"),
    pytest.param(MODEL_GEMINI_FLASH, id="judge+aggregator:gemini-2.5-flash"),
    pytest.param(MODEL_OPUS, id="escalation:openrouter-claude-opus-4.6"),
]


@pytest.mark.live
class TestEachModel:
    @pytest.mark.parametrize("model", ALL_CHAT_MODELS)
    async def test_completion(self, model):
        # max_tokens=200: reasoning models (Gemini 2.5 Flash, gpt-oss-120b) burn tokens
        # on internal thinking before emitting visible content; too-small budgets return None.
        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Say the word 'OK' and nothing else."}],
            max_tokens=200,
        )
        content = resp.choices[0].message.content
        assert content and len(content.strip()) > 0, f"empty response from {model}"

    async def test_embedding_model(self):
        resp = await litellm.aembedding(model=MODEL_EMBED, input=["hello world"])
        vec = resp.data[0]["embedding"]
        assert isinstance(vec, list) and len(vec) > 100, f"unexpected embedding shape: {len(vec)}"

    async def test_judge_and_aggregator_share_model(self):
        # Sanity — both special-purpose roles should resolve to the same Gemini model
        assert JUDGE_MODEL == AGGREGATOR_MODEL == MODEL_GEMINI_FLASH


# ---------------------------------------------------------------------------
# Section 3 — Component-level tests (real LLM calls, isolated nodes)
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestClassifier:
    async def test_classifier_self_answers_greeting(self):
        out = await classifier_node({"query": "hello"})
        assert out["can_self_answer"] is True
        assert out.get("final_response", "").strip() != ""

    async def test_classifier_marks_medical_critical(self):
        out = await classifier_node({"query": "Is it safe to combine metformin and alcohol?"})
        assert out["is_critical"] is True
        assert out["can_self_answer"] is False

    async def test_classifier_skips_critical_for_code(self):
        out = await classifier_node({"query": "Write JWT auth middleware for Express.js with refresh tokens."})
        assert out["is_critical"] is False  # code-context guardrail


@pytest.mark.live
class TestKNNRouter:
    """Reuses the module-level KNN index that _warm_knn_index built."""

    async def test_index_built(self):
        assert knn_mod.KNN_INDEX is not None
        assert knn_mod.KNN_INDEX["all_vectors"].shape[0] >= 100
        assert len(knn_mod.KNN_INDEX["all_labels"]) == knn_mod.KNN_INDEX["all_vectors"].shape[0]

    async def test_routes_simple_qa_to_llama_groq(self):
        best, _ = await semantic_route("What is the capital of Japan?", knn_mod.KNN_INDEX)
        assert best == MODEL_LLAMA_GROQ

    async def test_routes_math_to_gemini(self):
        best, _ = await semantic_route("What is the square root of 144?", knn_mod.KNN_INDEX)
        assert best == MODEL_GEMINI_FLASH

    async def test_routes_research_to_qwen(self):
        best, _ = await semantic_route(
            "Provide a comprehensive analysis of the Industrial Revolution.", knn_mod.KNN_INDEX,
        )
        assert best == MODEL_QWEN_235B


@pytest.mark.live
class TestWorker:
    async def test_worker_success(self):
        state = {"selected_models": [MODEL_LLAMA_GROQ], "query": "Reply with just: OK", "total_cost": 0.0, "total_latency": 0.0}
        out = await worker_node(state)
        wr = out["worker_responses"][0]
        assert wr["error"] is False
        assert wr["response"].strip() != ""

    async def test_worker_failure_does_not_leak_error_into_response(self):
        # Use a deliberately bogus model name — must fail cleanly
        state = {"selected_models": ["groq/this-model-definitely-does-not-exist"], "query": "hi", "total_cost": 0.0, "total_latency": 0.0}
        out = await worker_node(state)
        wr = out["worker_responses"][0]
        assert wr["error"] is True
        assert wr["response"] == ""  # error must NOT leak into user-visible field
        assert wr["error_msg"]  # but is captured for telemetry / trace

    async def test_parallel_worker_marks_failures_individually(self):
        state = {
            "selected_models": [MODEL_LLAMA_GROQ, "groq/nonexistent-model-xyz"],
            "subtasks": ["greet me", "respond ok"],
            "query": "ignored",
            "total_cost": 0.0,
            "total_latency": 0.0,
        }
        out = await parallel_worker_node(state)
        responses = out["worker_responses"]
        assert len(responses) == 2
        assert responses[0]["error"] is False
        assert responses[1]["error"] is True
        assert responses[1]["response"] == ""


class TestAggregator:
    async def test_aggregator_drops_all_failed(self):
        state = {
            "query": "anything",
            "worker_responses": [
                {"model": "x", "response": "", "error": True, "error_msg": "boom"},
                {"model": "y", "response": "", "error": True, "error_msg": "kaboom"},
            ],
            "subtasks": ["a", "b"],
            "total_cost": 0.0,
            "total_latency": 0.0,
        }
        out = await aggregator_node(state)
        # No LLM call, no error narration, empty aggregated_response
        assert out["aggregated_response"] == ""
        assert out["trace"][0]["action"] == "skipped"

    async def test_aggregator_passthrough_single_survivor(self):
        state = {
            "query": "anything",
            "worker_responses": [
                {"model": "x", "response": "", "error": True, "error_msg": "fail"},
                {"model": "y", "response": "the actual answer", "error": False},
            ],
            "subtasks": ["a", "b"],
            "total_cost": 0.0,
            "total_latency": 0.0,
        }
        out = await aggregator_node(state)
        assert out["aggregated_response"] == "the actual answer"
        assert out["trace"][0]["action"] == "passthrough"


@pytest.mark.live
class TestAggregatorLive:
    async def test_aggregator_merges_multiple_successes(self):
        state = {
            "query": "What are pros and cons of microservices?",
            "worker_responses": [
                {"model": "a", "response": "Pro: independent deployability.", "error": False},
                {"model": "b", "response": "Con: operational complexity.", "error": False},
            ],
            "subtasks": ["pros", "cons"],
            "total_cost": 0.0,
            "total_latency": 0.0,
        }
        out = await aggregator_node(state)
        merged = out["aggregated_response"]
        assert merged and len(merged) > 30  # actual merged content
        assert "agent" not in merged.lower()  # must not narrate agents
        assert "error" not in merged.lower()  # must not narrate errors


@pytest.mark.live
class TestJudge:
    async def test_judge_passes_reasonable_answer(self):
        state = {
            "query": "What is 2 + 2?",
            "worker_responses": [{"model": "x", "response": "2 + 2 equals 4.", "error": False}],
            "is_critical": False,
            "total_cost": 0.0,
            "total_latency": 0.0,
        }
        out = await judge_node(state)
        assert out["judge_score"] >= 5.0  # lenient mode for simple Q

    async def test_judge_rejects_empty_answer(self):
        state = {
            "query": "Explain TCP/IP in detail.",
            "worker_responses": [{"model": "x", "response": "I don't know.", "error": False}],
            "is_critical": True,  # forces strict mode
            "total_cost": 0.0,
            "total_latency": 0.0,
        }
        out = await judge_node(state)
        assert out["judge_score"] < 7.0


# ---------------------------------------------------------------------------
# Section 4 — Full LangGraph end-to-end (every routing path)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _warm_knn_index():
    """Build the KNN index once for the whole test session.
    Sync wrapper around the async builder — keeps pytest-asyncio scope rules out of it.
    """
    if not os.getenv("FIREWORKS_API_KEY"):
        return
    if knn_mod.KNN_INDEX is None:
        knn_mod.KNN_INDEX = asyncio.run(build_knn_index())


async def _run_graph(query: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "query": query,
        "trace": [],
        "worker_responses": [],
        "knn_scores": {},
        "selected_models": [],
        "total_cost": 0.0,
        "total_latency": 0.0,
        "escalation_count": 0,
    }
    final_state = None
    async for event in switchboard_graph.astream(initial, config=config, stream_mode="values"):
        final_state = event
    return final_state or {}


def _is_clean_answer(text: str) -> bool:
    """User-facing answer must contain real content, no error narration, no metrics footer."""
    if not text or not text.strip():
        return False
    bad_phrases = [
        "litellm.notfounderror",
        "cerebrasexception",
        "groqexception",
        "openrouterexception",
        "Error: ",
        "Cost: $",
        "Routed:",
        "Used:",
        "Saved vs",
        "Multiple agents encountered",
    ]
    return not any(p.lower() in text.lower() for p in bad_phrases)


@pytest.mark.live
class TestGraphPaths:
    async def test_graph_greeting_self_answer(self):
        state = await _run_graph("hello", "graph-greeting")
        assert _is_clean_answer(state.get("final_response", ""))

    async def test_graph_simple_qa(self):
        state = await _run_graph("What is the capital of Japan?", "graph-simple")
        assert _is_clean_answer(state["final_response"])
        assert "tokyo" in state["final_response"].lower()
        # If the classifier splits into subtasks the same model can repeat — only the routing target matters.
        assert set(state["selected_models"]) == {MODEL_LLAMA_GROQ}

    async def test_graph_math(self):
        state = await _run_graph("What is the square root of 144?", "graph-math")
        assert _is_clean_answer(state["final_response"])
        assert "12" in state["final_response"]
        assert set(state["selected_models"]) == {MODEL_GEMINI_FLASH}

    async def test_graph_code(self):
        state = await _run_graph("Write a Python function to reverse a linked list.", "graph-code")
        assert _is_clean_answer(state["final_response"])
        assert "def " in state["final_response"]
        assert set(state["selected_models"]) == {MODEL_KIMI_K2}

    async def test_graph_general_medium(self):
        state = await _run_graph("Explain the CAP theorem and its implications for databases.", "graph-general")
        assert _is_clean_answer(state["final_response"])
        assert set(state["selected_models"]) == {MODEL_GPT_OSS}

    async def test_graph_research(self):
        state = await _run_graph(
            "Summarize the economic impact of the Industrial Revolution on Europe.", "graph-research",
        )
        assert _is_clean_answer(state["final_response"])
        assert set(state["selected_models"]) == {MODEL_QWEN_235B}

    async def test_graph_critical_runs_judge(self):
        state = await _run_graph("What are the warning signs of a stroke?", "graph-critical")
        assert _is_clean_answer(state["final_response"])
        assert state.get("is_critical") is True
        assert "judge_score" in state  # judge ran

    async def test_graph_parallel_subtasks(self):
        state = await _run_graph(
            "Compare microservices vs monolith and list pros and cons of each.", "graph-parallel",
        )
        assert _is_clean_answer(state["final_response"])
        assert len(state.get("worker_responses", [])) >= 2  # parallel fan-out happened


# ---------------------------------------------------------------------------
# Section 5 — FastAPI endpoints (in-process, no uvicorn)
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestAPI:
    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def test_health(self, client):
        async with client as c:
            r = await c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["models"] == 7

    async def test_models(self, client):
        async with client as c:
            r = await c.get("/models")
        assert r.status_code == 200
        body = r.json()
        assert MODEL_LLAMA_GROQ in body["prototypes"]
        assert MODEL_KIMI_K2 in body["prototypes"]
        assert MODEL_GPT_OSS in body["prototypes"]

    async def test_chat_streams_clean_answer(self, client):
        """The /chat SSE stream must deliver a clean answer with no error narration."""
        async with client as c:
            r = await c.post(
                "/chat",
                json={"query": "What is the capital of Japan?", "session_id": "api-clean"},
                timeout=60.0,
            )
        assert r.status_code == 200

        final_payload = None
        for line in r.text.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if data.get("type") == "final":
                final_payload = data
        assert final_payload is not None, "no final SSE event"
        answer = final_payload["response"]
        assert _is_clean_answer(answer)
        assert "tokyo" in answer.lower()
        # Metrics live in payload (sidebar consumes them) — not in the answer text
        assert "Cost:" not in answer
        assert "Routed:" not in answer
        assert isinstance(final_payload["routed_models"], list)
        assert isinstance(final_payload["used_models"], list)
