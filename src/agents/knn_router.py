import time
import numpy as np
from collections import Counter, OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
import litellm

from core.state import SwitchboardState, TraceEntry
from core.config import MODEL_EMBED, KNN_K_VALUE
from core.prototypes import MODEL_PROTOTYPES

# Module-level KNN index — set once at FastAPI startup
KNN_INDEX = None

# Simple LRU cache for query embeddings to avoid redundant API calls
_EMBED_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_EMBED_CACHE_MAX = 1000


async def _cached_embed(query: str) -> list[float]:
    """Embed a query with LRU caching."""
    if query in _EMBED_CACHE:
        _EMBED_CACHE.move_to_end(query)
        return _EMBED_CACHE[query]
    response = await litellm.aembedding(model=MODEL_EMBED, input=[query])
    vec = response.data[0]["embedding"]
    _EMBED_CACHE[query] = vec
    if len(_EMBED_CACHE) > _EMBED_CACHE_MAX:
        _EMBED_CACHE.popitem(last=False)
    return vec


async def build_knn_index() -> dict:
    """Build the KNN index by embedding all prototype queries.
    Called ONCE at FastAPI startup. Stored in memory.
    """
    all_vectors = []
    all_labels = []

    for model_name, examples in MODEL_PROTOTYPES.items():
        # Embed all examples for this model in one batch
        response = await litellm.aembedding(model=MODEL_EMBED, input=examples)
        for i, item in enumerate(response.data):
            all_vectors.append(item["embedding"])
            all_labels.append(model_name)

    return {
        "all_vectors": np.array(all_vectors),
        "all_labels": all_labels,
    }


async def semantic_route(query: str, index: dict) -> tuple:
    """Embed a query and find the best model via top-5 KNN voting.

    Returns:
        (best_model, knn_scores) where knn_scores is {model: float}
    """
    # Embed the query (with LRU cache)
    query_vec = await _cached_embed(query)

    # Cosine similarity vs all prototypes
    scores = cosine_similarity([query_vec], index["all_vectors"])[0]

    # Top-5 nearest neighbors
    top5_idx = scores.argsort()[-5:][::-1]
    top5_models = [index["all_labels"][i] for i in top5_idx]

    # Majority vote
    best_model = Counter(top5_models).most_common(1)[0][0]

    # Build knn_scores dict for UI bar chart
    knn_scores = {}
    for idx in top5_idx:
        label = index["all_labels"][idx]
        score = float(scores[idx])
        if label not in knn_scores or score > knn_scores[label]:
            knn_scores[label] = score

    return best_model, knn_scores


async def knn_router_node(state: SwitchboardState) -> dict:
    """LangGraph node: route query to best model via KNN similarity."""
    global KNN_INDEX
    if KNN_INDEX is None:
        return {
            "error": "KNN index not initialized",
            "trace": [{"node": "knn_router", "action": "error",
                       "detail": "KNN index not built", "timestamp": time.time()}],
        }

    query_to_use = state.get("enriched_query") or state.get("query", "")
    subtasks = state.get("subtasks", [])
    embed_cost = 0.0

    if subtasks and len(subtasks) > 0:
        # Route each subtask separately
        selected_models = []
        combined_knn_scores = {}
        for subtask in subtasks:
            model, scores = await semantic_route(subtask, KNN_INDEX)
            selected_models.append(model)
            combined_knn_scores.update(scores)
        knn_scores = combined_knn_scores
    else:
        # Route the full query once
        best_model, knn_scores = await semantic_route(query_to_use, KNN_INDEX)
        selected_models = [best_model]

    # Estimate embedding cost (~$0.00001 per query)
    embed_cost = 0.00001 * (len(subtasks) if subtasks else 1)

    top_score = max(knn_scores.values()) if knn_scores else 0.0
    route_preview = ", ".join(selected_models[:4])
    if len(selected_models) > 4:
        route_preview += ", ..."

    trace_entry: TraceEntry = {
        "node": "knn_router",
        "action": "routed",
        "detail": f"models=[{route_preview}] top_score={top_score:.3f}",
        "timestamp": time.time(),
    }

    return {
        "selected_models": selected_models,
        "knn_scores": knn_scores,
        "trace": [trace_entry],
        "total_cost": state.get("total_cost", 0.0) + embed_cost,
        "total_latency": state.get("total_latency", 0.0), # Embeddings are near-instant
    }
