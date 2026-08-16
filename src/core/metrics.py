import litellm
from core.config import MODEL_COSTS

def calculate_cost(model: str, response: any) -> float:
    """Calculates cost using LiteLLM with a fallback to manual calculation if litellm returns 0."""
    try:
        cost = litellm.completion_cost(completion_response=response)
        if cost and cost > 0:
            return float(cost)
    except Exception:
        pass

    # Fallback to manual calculation based on config
    if not hasattr(response, "usage") or not response.usage:
        return 0.0
    
    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0)
    completion_tokens = getattr(usage, "completion_tokens", 0)
    
    # Clean model string to match config (sometimes provider/ is prefixed)
    cost_config = None
    for m, costs in MODEL_COSTS.items():
        if model == m or model.endswith(m) or m.endswith(model):
            cost_config = costs
            break
            
    if cost_config:
        input_cost = (prompt_tokens / 1_000_000) * cost_config["input"]
        output_cost = (completion_tokens / 1_000_000) * cost_config["output"]
        return input_cost + output_cost
        
    return 0.0
