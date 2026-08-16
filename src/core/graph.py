from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import SwitchboardState
from agents.classifier import classifier_node
from agents.knn_router import knn_router_node
from agents.worker import worker_node, parallel_worker_node
from agents.hitl import hitl_node
from agents.aggregator import aggregator_node
from agents.judge import judge_node, escalation_worker_node
from core.config import MAX_ESCALATIONS

def set_final(state: SwitchboardState):
    """Set the final response before graph exit and ensure metrics are preserved.
    Worker error strings are never leaked to the user; only successful responses are surfaced.
    If nothing succeeded, a clean fallback message is shown.
    """
    final_existing = state.get("final_response")
    if final_existing:
        res = final_existing
    else:
        agg = state.get("aggregated_response")
        if agg and agg.strip():
            res = agg
        else:
            workers = state.get("worker_responses", []) or []
            successful = [
                w for w in workers
                if not w.get("error") and (w.get("response") or "").strip()
            ]
            if successful:
                res = successful[-1].get("response", "")
            else:
                res = "Sorry, I couldn't generate a response right now. Please try again."

    return {
        "final_response": res,
        "total_cost": state.get("total_cost", 0.0),
        "total_latency": state.get("total_latency", 0.0),
        "selected_models": state.get("selected_models", []),
        "worker_responses": state.get("worker_responses", []),
        "escalation_count": state.get("escalation_count", 0),
        "escalation_model": state.get("escalation_model"),
    }

def route_from_classifier(state: SwitchboardState):
    """Determine path after classification."""
    if state.get("can_self_answer", False):
        return END
    if state.get("is_ambiguous", False):
        return "hitl"
    return "knn_router"

def route_from_knn(state: SwitchboardState):
    """Determine path after routing."""
    subtasks = state.get("subtasks", [])
    if len(subtasks) > 0:
        return "parallel_worker"
    return "worker"

def route_from_worker(state: SwitchboardState):
    """Determine path after a single worker."""
    if state.get("is_critical", False):
        return "judge"
    return "set_final"

def route_from_aggregator(state: SwitchboardState):
    """Determine path after aggregating multiple workers."""
    if state.get("is_critical", False):
        return "judge"
    return "set_final"

def route_from_judge(state: SwitchboardState):
    """Determine path after judge evaluates quality."""
    score = state.get("judge_score", 0.0)
    # Check if failed AND we haven't hit the escalation limit
    # The judge component sets escalation_model if it failed
    if state.get("escalation_model") and state.get("escalation_count", 0) < MAX_ESCALATIONS:
        return "escalation_worker"
    return "set_final"

def create_graph():
    """LangGraph pipeline definition."""
    workflow = StateGraph(SwitchboardState)
    
    # Add Nodes
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("hitl", hitl_node)
    workflow.add_node("knn_router", knn_router_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("parallel_worker", parallel_worker_node)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("escalation_worker", escalation_worker_node)
    workflow.add_node("set_final", set_final)
    
    # Define Edges / Routing
    workflow.set_entry_point("classifier")
    
    workflow.add_conditional_edges(
        "classifier",
        route_from_classifier,
        {END: END, "hitl": "hitl", "knn_router": "knn_router"}
    )
    
    # After HITL clarifies -> we route it
    workflow.add_edge("hitl", "knn_router")
    
    workflow.add_conditional_edges(
        "knn_router",
        route_from_knn,
        {"parallel_worker": "parallel_worker", "worker": "worker"}
    )
    
    workflow.add_conditional_edges(
        "worker",
        route_from_worker,
        {"judge": "judge", "set_final": "set_final"}
    )
    
    # Parallel workers always go to an aggregator
    workflow.add_edge("parallel_worker", "aggregator")
    
    workflow.add_conditional_edges(
        "aggregator",
        route_from_aggregator,
        {"judge": "judge", "set_final": "set_final"}
    )
    
    workflow.add_conditional_edges(
        "judge",
        route_from_judge,
        {"escalation_worker": "escalation_worker", "set_final": "set_final"}
    )
    
    workflow.add_edge("escalation_worker", "set_final")
    workflow.add_edge("set_final", END)
    
    # Setup persistence memory for HITL interrupts and state continuity
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

# Export a default instance
switchboard_graph = create_graph()


if __name__ == "__main__":
    import asyncio

    async def test():
        test_queries = [
            "hello",                                              # should self-answer
            "write quicksort in python",                         # should route to Kimi K2
            "explain WWI causes and its economic impact",        # should trigger parallel subtasks
        ]

        for i, q in enumerate(test_queries):
            print(f"\n{'='*60}")
            print(f"TEST {i+1}: {q}")
            print("="*60)
            config = {"configurable": {"thread_id": f"test-{i}"}}
            final_state = None
            async for event in switchboard_graph.astream({"query": q}, config=config, stream_mode="values"):
                final_state = event

            if final_state:
                print(f"\nFinal Response: {str(final_state.get('final_response', ''))[:200]}")
                print(f"KNN Scores: {final_state.get('knn_scores', {})}")
                print(f"Selected Models: {final_state.get('selected_models', [])}")
                print(f"Total Cost: ${final_state.get('total_cost', 0):.6f}")
                print(f"Trace:")
                for t in final_state.get("trace", []):
                    print(f"  [{t['node']}] {t['action']}: {t.get('detail', '')}")

    asyncio.run(test())
