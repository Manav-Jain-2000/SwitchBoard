import time
import asyncio
import litellm
from core.state import SwitchboardState, TraceEntry
from core.metrics import calculate_cost


async def worker_node(state: SwitchboardState) -> dict:
    """Single worker: calls the KNN-selected model with timeout and cost tracking."""
    selected_models = state.get("selected_models", [])
    model = selected_models[0] if selected_models else "groq/llama-3.1-8b-instant"

    query = state.get("enriched_query") or state.get("query", "")

    start = time.time()
    error_msg = ""
    try:
        response = await asyncio.wait_for(
            litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": query}],
            ),
            timeout=30,
        )
        latency_ms = (time.time() - start) * 1000
        output_content = response.choices[0].message.content
        cost_usd = calculate_cost(model, response)

    except asyncio.TimeoutError:
        latency_ms = 30000.0
        output_content = ""
        error_msg = "timeout"
        cost_usd = 0.0

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        output_content = ""
        error_msg = str(e)
        cost_usd = 0.0

    worker_result = {
        "model": model,
        "response": output_content,
        "cost_usd": cost_usd,
        "latency_ms": round(latency_ms, 2),
        "error": bool(error_msg),
        "error_msg": error_msg,
    }

    action = "completed" if not error_msg else "failed"
    detail = (
        f"model={model} latency={latency_ms:.0f}ms cost=${cost_usd:.6f}"
        if not error_msg
        else f"model={model} error={error_msg[:120]}"
    )
    trace_entry: TraceEntry = {
        "node": "worker",
        "action": action,
        "detail": detail,
        "timestamp": time.time(),
    }

    return {
        "worker_responses": [worker_result],
        "trace": [trace_entry],
        "total_cost": state.get("total_cost", 0.0) + cost_usd,
        "total_latency": state.get("total_latency", 0.0) + (latency_ms / 1000),
    }


async def parallel_worker_node(state: SwitchboardState) -> dict:
    """Fan-out: one coroutine per subtask using the KNN-selected model for each."""
    query = state.get("enriched_query") or state.get("query", "")
    subtasks = state.get("subtasks", [])
    selected_models = state.get("selected_models", [])

    async def run_subtask(subtask: str, model: str) -> dict:
        start = time.time()
        err = ""
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"You are a specialist. Focus ONLY on this subtask: {subtask}"},
                        {"role": "user", "content": f"For query: {query}\nHandle this aspect: {subtask}"},
                    ],
                ),
                timeout=30,
            )
            latency_ms = (time.time() - start) * 1000
            content = response.choices[0].message.content
            cost_usd = calculate_cost(model, response)
        except asyncio.TimeoutError:
            latency_ms = 30000.0
            content = ""
            err = "timeout"
            cost_usd = 0.0
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            content = ""
            err = str(e)
            cost_usd = 0.0

        return {
            "model": model,
            "response": content,
            "cost_usd": cost_usd,
            "latency_ms": round(latency_ms, 2),
            "error": bool(err),
            "error_msg": err,
        }

    # Build coroutines — match each subtask to its selected model
    coroutines = []
    for i, subtask in enumerate(subtasks):
        model = selected_models[i] if i < len(selected_models) else selected_models[-1]
        coroutines.append(run_subtask(subtask, model))

    # Fan-out with asyncio.gather
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    worker_responses = []
    total_cost = 0.0
    total_latency = 0.0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            worker_responses.append({
                "model": selected_models[i] if i < len(selected_models) else "unknown",
                "response": "",
                "cost_usd": 0.0,
                "latency_ms": 0.0,
                "error": True,
                "error_msg": str(result),
            })
        else:
            worker_responses.append(result)
            total_cost += result.get("cost_usd", 0.0)
            total_latency = max(total_latency, result.get("latency_ms", 0.0) / 1000)

    trace_entry: TraceEntry = {
        "node": "parallel_workers",
        "action": "fan_out",
        "detail": f"{len(subtasks)} agents dispatched",
        "timestamp": time.time(),
    }

    return {
        "worker_responses": worker_responses,
        "trace": [trace_entry],
        "total_cost": state.get("total_cost", 0.0) + total_cost,
        "total_latency": state.get("total_latency", 0.0) + total_latency,
    }
