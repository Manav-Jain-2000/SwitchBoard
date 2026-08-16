import json
import re
import time
import litellm
from core.state import SwitchboardState, TraceEntry
from core.config import (
    JUDGE_MODEL, JUDGE_THRESHOLD, JUDGE_THRESHOLDS,
    MAX_ESCALATIONS, MODEL_OPUS, ESCALATION_MODELS,
)
from core.metrics import calculate_cost

_CODE_KEYWORDS = frozenset([
    'write', 'implement', 'code', 'function', 'class', 'api', 'endpoint',
    'deploy', 'kubernetes', 'docker', 'pipeline', 'ci', 'cd', 'jwt', 'oauth',
    'middleware', 'schema', 'database', 'microservices', 'architecture',
    'redis', 'express', 'react', 'python', 'java', 'typescript', 'golang',
    'production', 'scalable', 'distributed', 'encryption', 'auth',
])


def _infer_category(query: str, is_critical: bool) -> str:
    """Infer query category for threshold/escalation model selection."""
    if not is_critical:
        return "default"
    tokens = set(re.sub(r"[^a-z\s]", "", query.lower()).split())
    if tokens & _CODE_KEYWORDS:
        return "critical_code"
    return "critical"


async def judge_node(state: SwitchboardState) -> dict:
    """Evaluate response quality and approve or trigger escalation."""

    query = state.get("enriched_query") or state.get("query", "")

    # Evaluate previously generated content — only surface successful worker output.
    agg = state.get("aggregated_response")
    if agg and agg.strip():
        response_to_evaluate = agg
    else:
        worker_responses = state.get("worker_responses", []) or []
        successful = [
            w for w in worker_responses
            if not w.get("error") and (w.get("response") or "").strip()
        ]
        if successful:
            response_to_evaluate = successful[-1].get("response", "")
        else:
            response_to_evaluate = "No response generated."

    is_critical = state.get("is_critical", False)
    category = _infer_category(query, is_critical)
    threshold = JUDGE_THRESHOLDS.get(category, JUDGE_THRESHOLD)
    default_escalation_model = ESCALATION_MODELS.get(category, ESCALATION_MODELS["default"])

    strictness = "STRICT" if is_critical else "LENIENT"
    context = (
        "This is a CRITICAL query (medical/legal/financial). Be strict — accuracy is paramount. "
        "Only pass responses that are thorough, accurate, and safe."
        if is_critical else
        "This is a standard query. Be lenient — if the response reasonably answers the question, "
        "give it a passing score. Minor gaps are acceptable. Only reject clearly wrong or empty responses."
    )

    prompt = f"""Evaluate the agent's response. Mode: {strictness}.
{context}

Return ONLY JSON with these keys:
- score: float (0-10). For standard queries, a reasonable answer should score 7+. For critical queries, only thorough answers score 7+.
- dimensions: dict with 'accuracy', 'completeness', 'reasoning_depth' (each 0-10)
- failure_reason: string (why it failed, or empty if it passes)
- retry_instruction: string (how to improve, or empty)
- escalate_to: string (model for retry, or empty)

Original Query: {query}
Agent Response: {response_to_evaluate}
"""

    try:
        start = time.time()
        response = await litellm.acompletion(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": f"You are a response quality judge. Score generously for standard queries (pass if the answer is useful). Score strictly for critical queries (medical/legal/financial). Threshold: {threshold}. If score < threshold, provide failure_reason and retry_instruction."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        latency_ms = (time.time() - start) * 1000
        content = response.choices[0].message.content
        result = json.loads(content)
        cost = calculate_cost(JUDGE_MODEL, response)

    except Exception as e:
        # Failsafe fallback
        result = {
            "score": 0.0,
            "failure_reason": f"Evaluation system error: {str(e)}",
            "retry_instruction": "Retry with a highly intelligent model.",
            "escalate_to": default_escalation_model,
        }
        cost = 0.0
        latency_ms = 0.0

    score = result.get("score", 0.0)
    passed = score >= threshold

    trace_entry: TraceEntry = {
        "node": "judge",
        "action": "approved" if passed else "rejected",
        "detail": f"Score {score:.1f}/{threshold}. " + (f"Passed." if passed else f"Reason: {result.get('failure_reason')}"),
        "timestamp": time.time()
    }

    output = {
        "judge_score": score,
        "judge_feedback": result.get("failure_reason", ""),
        "trace": [trace_entry],
        "total_cost": state.get("total_cost", 0.0) + cost,
        "total_latency": state.get("total_latency", 0.0) + (latency_ms / 1000),
    }

    if passed:
        return output
    else:
        # Use category-appropriate escalation model instead of always Opus
        escalation_model = result.get("escalate_to") or default_escalation_model
        output["escalation_model"] = escalation_model
        output["escalation_instruction"] = result.get("retry_instruction", "")
        return output

async def escalation_worker_node(state: SwitchboardState) -> dict:
    """Invoked when the judge fails a response. Uses designated more capable model."""
    
    target_query = state.get("enriched_query") or state.get("query", "")
    escalation_instruction = state.get("escalation_instruction", "")
    escalation_model = state.get("escalation_model", MODEL_OPUS)
    
    prompt = f"Previous attempt failed: {escalation_instruction}. Fix this specifically and address the query below.\n\nQuery: {target_query}"
    
    try:
        start = time.time()
        response = await litellm.acompletion(
            model=escalation_model,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.time() - start) * 1000
        output_content = response.choices[0].message.content
        actual_model = response.model if hasattr(response, 'model') else escalation_model
        cost = calculate_cost(actual_model, response)

    except Exception as e:
        output_content = f"Escalation failed entirely: {str(e)}"
        actual_model = escalation_model
        cost = 0.0
        latency_ms = 0.0
        
    trace_entry: TraceEntry = {
        "node": "escalation_worker",
        "action": "escalated_response",
        "detail": f"Used {actual_model} after judge rejection.",
        "timestamp": time.time()
    }
    
    return {
        "final_response": output_content,
        "escalation_count": state.get("escalation_count", 0) + 1,
        "trace": [trace_entry],
        "total_cost": state.get("total_cost", 0.0) + cost,
        "total_latency": state.get("total_latency", 0.0) + (latency_ms / 1000),
    }
