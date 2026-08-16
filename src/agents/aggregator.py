import time
import litellm
from core.state import SwitchboardState, TraceEntry
from core.config import AGGREGATOR_MODEL
from core.metrics import calculate_cost


async def aggregator_node(state: SwitchboardState) -> dict:
    """Merges multiple parallel worker outputs into a single cohesive response.
    Failed workers are dropped — they are not narrated into the user-visible answer.
    """
    query = state.get("enriched_query") or state.get("query", "")
    worker_responses = state.get("worker_responses", [])
    subtasks = state.get("subtasks", [])

    # Drop failed workers and any with empty responses; their errors live in the trace, not the answer.
    successful = [
        (i, w) for i, w in enumerate(worker_responses)
        if not w.get("error") and (w.get("response") or "").strip()
    ]

    if not successful:
        # All workers failed — return empty so set_final / escalation can supply a clean fallback.
        trace_entry: TraceEntry = {
            "node": "aggregator",
            "action": "skipped",
            "detail": f"All {len(worker_responses)} workers failed; nothing to merge",
            "timestamp": time.time(),
        }
        return {
            "aggregated_response": "",
            "trace": [trace_entry],
            "total_cost": state.get("total_cost", 0.0),
            "total_latency": state.get("total_latency", 0.0),
        }

    if len(successful) == 1:
        # Single survivor — return verbatim, no LLM merge call needed.
        only = successful[0][1]
        trace_entry = {
            "node": "aggregator",
            "action": "passthrough",
            "detail": f"1 of {len(worker_responses)} workers succeeded; passed response through",
            "timestamp": time.time(),
        }
        return {
            "aggregated_response": only.get("response", ""),
            "trace": [trace_entry],
            "total_cost": state.get("total_cost", 0.0),
            "total_latency": state.get("total_latency", 0.0),
        }

    parts = []
    for orig_i, w in successful:
        label = subtasks[orig_i] if orig_i < len(subtasks) else f"Agent {orig_i + 1}"
        parts.append(f"[{label}]: {w.get('response', '')}")
    combined_context = "\n\n".join(parts)

    try:
        start = time.time()
        response = await litellm.acompletion(
            model=AGGREGATOR_MODEL,
            messages=[
                {"role": "system", "content": "Merge these agent responses into one cohesive answer for the user. Do not mention agents, errors, or models. No redundancy. Preserve all insights."},
                {"role": "user", "content": f"Query: {query}\n\nAgent responses:\n{combined_context}"},
            ],
        )
        latency_ms = (time.time() - start) * 1000
        aggregated_content = response.choices[0].message.content
        cost = calculate_cost(AGGREGATOR_MODEL, response)

    except Exception:
        # Aggregation LLM failed — fall back to concatenation of successful responses (no error narration).
        aggregated_content = "\n\n".join(w.get("response", "") for _, w in successful)
        cost = 0.0
        latency_ms = 0.0

    trace_entry = {
        "node": "aggregator",
        "action": "merged",
        "detail": f"Merged {len(successful)} of {len(worker_responses)} responses using {AGGREGATOR_MODEL}",
        "timestamp": time.time(),
    }

    return {
        "aggregated_response": aggregated_content,
        "trace": [trace_entry],
        "total_cost": state.get("total_cost", 0.0) + cost,
        "total_latency": state.get("total_latency", 0.0) + (latency_ms / 1000),
    }
