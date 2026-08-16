from typing import TypedDict, List, Annotated, Dict, Any, Optional
from operator import add

class TraceEntry(TypedDict):
    node: str
    action: str
    detail: Optional[str]
    timestamp: float

class SwitchboardState(TypedDict):
    """SwitchboardState TypedDict for LangGraph state management."""
    # Query
    query: str
    enriched_query: str
    original_query: str

    # Classification flags
    can_self_answer: bool
    is_critical: bool
    is_ambiguous: bool

    # Clarification
    clarifying_question: str
    user_clarification: str
    conversation_turns: int

    # Routing
    subtasks: list[str]
    selected_models: list[str]

    # Worker related
    worker_responses: Annotated[List[Dict[str, Any]], add]
    aggregated_response: str
    final_response: str

    # Judge
    judge_score: float
    judge_feedback: str

    # Escalation
    escalation_model: str
    escalation_instruction: str
    escalation_count: int

    # Metrics & Trace
    knn_scores: Dict[str, float]
    trace: Annotated[List[TraceEntry], add]
    total_cost: float
    total_latency: float

    # Error
    error: Optional[str]
