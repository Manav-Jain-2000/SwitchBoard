import time
from langgraph.types import interrupt
from core.state import SwitchboardState, TraceEntry


def hitl_node(state: SwitchboardState) -> dict:
    """Human-in-the-loop interrupt node.
    Sync function — LangGraph's interrupt() suspends execution until resumed.
    Uses the classifier's clarifying_question if available.
    """
    # Use classifier's clarifying question, or generate a fallback
    question = state.get("clarifying_question", "")
    if not question:
        query = state.get("query", "")
        question = f"Your query '{query}' is ambiguous. Could you please provide more details?"

    # LangGraph interrupt — pauses the graph
    user_answer = interrupt({"question": question})

    clarification = str(user_answer) if user_answer else "please proceed with best guess"
    enriched = state.get("query", "") + " [Clarified: " + clarification + "]"

    trace_entry: TraceEntry = {
        "node": "hitl",
        "action": "resumed",
        "detail": clarification[:80],
        "timestamp": time.time(),
    }

    return {
        "clarifying_question": question,
        "user_clarification": clarification,
        "enriched_query": enriched,
        "conversation_turns": state.get("conversation_turns", 0) + 1,
        "trace": [trace_entry],
        "total_cost": state.get("total_cost", 0.0),
        "total_latency": state.get("total_latency", 0.0),
    }
