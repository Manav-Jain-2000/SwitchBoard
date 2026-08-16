import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command

from core.graph import switchboard_graph
from core.config import MODEL_COSTS, GPT5_BASELINE_COST
from core.prototypes import MODEL_PROTOTYPES
from core.logging import get_logger
import agents.knn_router as knn_mod

log = get_logger("api")

active_sessions = {}


class ChatRequest(BaseModel):
    query: str
    session_id: str


class ResumeRequest(BaseModel):
    session_id: str
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    from agents.knn_router import build_knn_index
    knn_mod.KNN_INDEX = await build_knn_index()
    log.info("KNN index built: %d vectors loaded", knn_mod.KNN_INDEX["all_vectors"].shape[0])
    yield


app = FastAPI(title="SwitchBoard", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


async def state_to_sse(generator):
    try:
        async for event in generator:
            kind = event.get("event")
            if kind == "on_chain_end" and not event.get("name") == "LangGraph":
                data = event.get("data", {})
                if "output" in data and isinstance(data["output"], dict):
                    output = data["output"]
                    if "trace" in output:
                        trace_list = output["trace"]
                        if trace_list:
                            latest_trace = trace_list[-1]
                            yield f"data: {json.dumps({'type': 'trace', 'entry': latest_trace, 'knn_scores': output.get('knn_scores', {})})}\n\n"
                    if "clarifying_question" in output and output.get("clarifying_question"):
                        yield f"data: {json.dumps({'type': 'interrupt', 'question': output['clarifying_question']})}\n\n"
                        break
                    if "final_response" in output and output.get("final_response"):
                        total_cost = output.get("total_cost", 0.0)
                        total_latency = output.get("total_latency", 0.0)
                        cost_saved = GPT5_BASELINE_COST - total_cost
                        routed_models = output.get("selected_models", [])
                        worker_responses = output.get("worker_responses", []) or []
                        used_models = [w.get("model", "unknown") for w in worker_responses if isinstance(w, dict)]
                        if output.get("escalation_count", 0) > 0 and output.get("escalation_model"):
                            used_models.append(output["escalation_model"])
                        payload = {
                            "type": "final",
                            "response": output["final_response"],
                            "total_cost": round(total_cost, 6),
                            "total_latency": round(total_latency, 2),
                            "cost_saved": round(cost_saved, 6),
                            "baseline_model": "gpt-5",
                            "baseline_cost": round(GPT5_BASELINE_COST, 6),
                            "routed_models": routed_models,
                            "used_models": used_models,
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
    except Exception as e:
        log.error("SSE stream error: %s", e)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # Each new query gets a fresh thread_id so its checkpoint can't pick up stale
    # final_response / aggregated_response / worker_responses from the previous query.
    # active_sessions remembers the LATEST thread_id per session so /resume can find it for HITL.
    thread_id = f"{req.session_id}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    active_sessions[req.session_id] = config
    initial_state = {
        "query": req.query,
        "trace": [],
        "worker_responses": [],
        "knn_scores": {},
        "selected_models": [],
        "total_cost": 0.0,
        "total_latency": 0.0,
        "escalation_count": 0,
    }
    stream_generator = switchboard_graph.astream_events(initial_state, config=config, version="v2")
    return StreamingResponse(state_to_sse(stream_generator), media_type="text/event-stream")


@app.post("/resume")
async def resume_endpoint(req: ResumeRequest):
    config = active_sessions.get(req.session_id)
    if not config:
        return {"error": "Session not found"}
    stream_generator = switchboard_graph.astream_events(Command(resume=req.answer), config=config, version="v2")
    return StreamingResponse(state_to_sse(stream_generator), media_type="text/event-stream")


@app.get("/trace/{session_id}")
async def get_trace(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    state = switchboard_graph.get_state(config)
    if state and hasattr(state, "values"):
        return {"trace": state.values.get("trace", [])}
    return {"trace": []}


@app.get("/models")
async def get_models():
    return {
        "prototypes": list(MODEL_PROTOTYPES.keys()),
        "costs": MODEL_COSTS,
        "baseline_model": "gpt-5",
        "baseline_cost": GPT5_BASELINE_COST,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "models": 7,
        "knn_index_loaded": knn_mod.KNN_INDEX is not None,
    }
