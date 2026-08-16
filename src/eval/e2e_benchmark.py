import asyncio
import argparse
import csv
import json
import os
import time
import uuid
from datetime import datetime

from langgraph.types import Command

from core.config import GPT5_BASELINE_COST, CATEGORY_BASELINES
from core.graph import switchboard_graph
from eval.benchmark import QUERIES
from eval.quality_scorer import score_response
import agents.knn_router as knn_mod
from agents.knn_router import build_knn_index


def _extract_used_models(memory: dict) -> list[str]:
    used = []
    for item in memory.get("worker_responses", []) or []:
        if isinstance(item, dict) and item.get("model"):
            used.append(item["model"])
    if memory.get("escalation_count", 0) > 0 and memory.get("escalation_model"):
        used.append(memory["escalation_model"])
    return used


def _extract_flow(memory: dict) -> list[str]:
    flow = []
    for trace in memory.get("trace", []) or []:
        node = trace.get("node")
        if node:
            flow.append(node)
    return flow


async def _run_single_query(idx: int, query: str, expected: str, category: str, query_timeout_s: float) -> dict:
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    initial_state = {
        "query": query,
        "trace": [],
        "worker_responses": [],
        "knn_scores": {},
        "selected_models": [],
        "total_cost": 0.0,
        "total_latency": 0.0,
        "escalation_count": 0,
    }

    start = time.time()
    error = ""
    try:
        async def _execute_query():
            async for _ in switchboard_graph.astream(initial_state, config=config, stream_mode="values"):
                pass

            state_snapshot = switchboard_graph.get_state(config)
            if state_snapshot.next and "hitl" in state_snapshot.next:
                async for _ in switchboard_graph.astream(
                    Command(resume="please proceed with best guess"),
                    config=config,
                    stream_mode="values",
                ):
                    pass

            return switchboard_graph.get_state(config).values

        memory = await asyncio.wait_for(_execute_query(), timeout=query_timeout_s)
    except asyncio.TimeoutError:
        memory = {}
        error = f"query_timeout>{query_timeout_s}s"
    except Exception as exc:
        memory = {}
        error = str(exc)

    latency = time.time() - start
    routed_models = memory.get("selected_models", []) or []
    used_models = _extract_used_models(memory)
    routed_model = routed_models[0] if routed_models else "unknown"
    cost = float(memory.get("total_cost", 0.0) or 0.0)
    top_knn = max((memory.get("knn_scores", {}) or {}).values(), default=0.0)

    flow_nodes = _extract_flow(memory)
    can_self_answer = bool(memory.get("can_self_answer", False))

    failure_type = ""
    if error.startswith("query_timeout>"):
        failure_type = "benchmark_timeout"
    elif error:
        failure_type = "runtime_error"
    elif routed_model == "unknown" and flow_nodes == ["classifier"] and can_self_answer:
        failure_type = "early_self_answer_exit"
    elif routed_model == "unknown":
        failure_type = "no_routing_output"

    # Quality scoring on final response
    final_response = str(memory.get("final_response", "") or "")
    is_critical = bool(memory.get("is_critical", False))
    quality = await score_response(query, final_response, is_critical=is_critical)

    result = {
        "id": idx,
        "query": query,
        "expected": expected,
        "routed_model": routed_model,
        "routed_models": routed_models,
        "used_models": used_models,
        "correct_routing": routed_model == expected,
        "critical": bool(memory.get("is_critical", False)),
        "escalated": (memory.get("escalation_count", 0) or 0) > 0,
        "latency_s": round(latency, 3),
        "graph_latency_s": round(float(memory.get("total_latency", 0.0) or 0.0), 3),
        "cost_usd": round(cost, 6),
        "saved_vs_gpt5_usd": round(GPT5_BASELINE_COST - cost, 6),
        "knn_top_score": round(float(top_knn), 4),
        "flow_nodes": flow_nodes,
        "can_self_answer": can_self_answer,
        "failure_type": failure_type,
        "error": error,
        "category": category,
        "baseline_cost": CATEGORY_BASELINES.get(category, GPT5_BASELINE_COST),
        "saved_vs_baseline": round(CATEGORY_BASELINES.get(category, GPT5_BASELINE_COST) - cost, 6),
        "success": error == "",
        "quality_score": quality["overall_score"],
        "quality_dimensions": quality["dimensions"],
        "quality_cost_usd": quality["cost_usd"],
    }
    return result


async def run_e2e_benchmark(limit: int | None = None, query_timeout_s: float = 90.0) -> tuple[dict, list[dict], str, str]:
    queries = QUERIES[:limit] if limit else QUERIES
    print(f"Running end-to-end benchmark: {len(queries)} queries", flush=True)

    if knn_mod.KNN_INDEX is None:
        print("Building KNN index...", flush=True)
        knn_mod.KNN_INDEX = await build_knn_index()

    results = []
    for idx, item in enumerate(queries, start=1):
        result = await _run_single_query(
            idx=idx,
            query=item["q"],
            category=item.get("category", "unknown"),
            expected=item["expected"],
            query_timeout_s=query_timeout_s,
        )
        status = "OK" if result["success"] else "FAIL"
        route_ok = "match" if result["correct_routing"] else "mismatch"
        print(
            f"[{idx}/{len(queries)}] {status} {route_ok} "
            f"lat={result['latency_s']:.2f}s routed={result['routed_model']} used={','.join(result['used_models']) or 'n/a'}"
            ,
            flush=True,
        )
        results.append(result)

    success_rows = [r for r in results if r["success"]]
    success_count = len(success_rows)
    failed_count = len(results) - success_count
    timeout_count = sum(1 for r in results if r.get("failure_type") == "benchmark_timeout")
    early_exit_count = sum(1 for r in results if r.get("failure_type") == "early_self_answer_exit")
    no_routing_count = sum(1 for r in results if r.get("failure_type") == "no_routing_output")
    total_cost = sum(r["cost_usd"] for r in success_rows)
    total_latency = sum(r["latency_s"] for r in success_rows)
    avg_latency = (total_latency / success_count) if success_count else 0.0
    avg_cost = (total_cost / success_count) if success_count else 0.0
    correct_routing = sum(1 for r in success_rows if r["correct_routing"])
    routing_accuracy = (correct_routing / success_count * 100.0) if success_count else 0.0
    baseline_total = GPT5_BASELINE_COST * success_count
    saved_total = baseline_total - total_cost

    # P50/P95 latency
    latencies = sorted(r["latency_s"] for r in success_rows)
    p50_latency = latencies[len(latencies) // 2] if latencies else 0.0
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1) if latencies else 0
    p95_latency = latencies[p95_idx] if latencies else 0.0

    # Escalation stats
    escalation_count = sum(1 for r in success_rows if r.get("escalated", False))
    escalation_rate = (escalation_count / success_count * 100) if success_count else 0.0

    # Confusion matrix: expected model -> {actual model: count}
    confusion_matrix: dict[str, dict[str, int]] = {}
    for r in success_rows:
        expected = r.get("expected", "unknown")
        actual = r.get("routed_model", "unknown")
        if expected not in confusion_matrix:
            confusion_matrix[expected] = {}
        confusion_matrix[expected][actual] = confusion_matrix[expected].get(actual, 0) + 1

    # Cost breakdown by pipeline stage
    stage_costs: dict[str, float] = {}
    for r in success_rows:
        for trace in r.get("flow_nodes", []):
            if trace not in stage_costs:
                stage_costs[trace] = 0.0
    # Approximate stage costs from trace data in results
    for r in success_rows:
        cost = r.get("cost_usd", 0.0)
        nodes = r.get("flow_nodes", [])
        if r.get("escalated", False):
            stage_costs["escalation"] = stage_costs.get("escalation", 0.0) + cost * 0.7
            stage_costs["judge"] = stage_costs.get("judge", 0.0) + cost * 0.1
            stage_costs["worker"] = stage_costs.get("worker", 0.0) + cost * 0.15
            stage_costs["classifier"] = stage_costs.get("classifier", 0.0) + cost * 0.05
        elif r.get("critical", False):
            stage_costs["judge"] = stage_costs.get("judge", 0.0) + cost * 0.2
            stage_costs["worker"] = stage_costs.get("worker", 0.0) + cost * 0.7
            stage_costs["classifier"] = stage_costs.get("classifier", 0.0) + cost * 0.1
        else:
            stage_costs["worker"] = stage_costs.get("worker", 0.0) + cost * 0.85
            stage_costs["classifier"] = stage_costs.get("classifier", 0.0) + cost * 0.15

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_queries": len(results),
        "successful_queries": success_count,
        "failed_queries": failed_count,
        "timeout_failures": timeout_count,
        "early_self_answer_exits": early_exit_count,
        "no_routing_output_cases": no_routing_count,
        "routing_accuracy_success_only_pct": round(routing_accuracy, 2),
        "avg_latency_s": round(avg_latency, 3),
        "p50_latency_s": round(p50_latency, 3),
        "p95_latency_s": round(p95_latency, 3),
        "avg_cost_usd": round(avg_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "gpt5_baseline_per_query_usd": GPT5_BASELINE_COST,
        "gpt5_baseline_total_usd_success_only": round(baseline_total, 6),
        "saved_vs_gpt5_total_usd_success_only": round(saved_total, 6),
        "escalation_count": escalation_count,
        "escalation_rate_pct": round(escalation_rate, 2),
        "confusion_matrix": confusion_matrix,
        "cost_breakdown_by_stage": {k: round(v, 6) for k, v in sorted(stage_costs.items())},
        "avg_quality_score": round(sum(r.get("quality_score", 0) for r in success_rows) / success_count, 2) if success_count else 0.0,
        "quality_scoring_cost_usd": round(sum(r.get("quality_cost_usd", 0) for r in success_rows), 6),
    }

    # Per-category breakdown
    categories = {}
    for r in success_rows:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"count": 0, "total_cost": 0.0, "total_baseline": 0.0, "correct": 0, "escalated": 0, "latencies": [], "quality_scores": []}
        categories[cat]["count"] += 1
        categories[cat]["total_cost"] += r["cost_usd"]
        categories[cat]["total_baseline"] += r.get("baseline_cost", GPT5_BASELINE_COST)
        categories[cat]["latencies"].append(r["latency_s"])
        categories[cat]["quality_scores"].append(r.get("quality_score", 0.0))
        if r["correct_routing"]:
            categories[cat]["correct"] += 1
        if r.get("escalated", False):
            categories[cat]["escalated"] += 1

    category_summary = {}
    for cat, data in sorted(categories.items()):
        avg_cost = data["total_cost"] / data["count"]
        avg_baseline = data["total_baseline"] / data["count"]
        savings_pct = ((avg_baseline - avg_cost) / avg_baseline * 100) if avg_baseline > 0 else 0
        cat_latencies = sorted(data["latencies"])
        cat_p50 = cat_latencies[len(cat_latencies) // 2] if cat_latencies else 0.0
        category_summary[cat] = {
            "queries": data["count"],
            "routing_accuracy": round(data["correct"] / data["count"] * 100, 1),
            "avg_cost": round(avg_cost, 6),
            "avg_baseline": round(avg_baseline, 6),
            "savings_pct": round(savings_pct, 1),
            "escalation_rate_pct": round(data["escalated"] / data["count"] * 100, 1),
            "p50_latency_s": round(cat_p50, 3),
            "avg_quality_score": round(sum(data["quality_scores"]) / data["count"], 2) if data["count"] else 0.0,
        }
    summary["per_category"] = category_summary

    os.makedirs("eval", exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join("eval", f"e2e_benchmark_{ts}.json")
    csv_path = os.path.join("eval", f"e2e_benchmark_{ts}.csv")
    latest_json = os.path.join("eval", "e2e_benchmark_latest.json")
    latest_csv = os.path.join("eval", "e2e_benchmark_latest.csv")

    payload = {"summary": summary, "results": results}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_headers = [
        "id",
        "query",
        "expected",
        "routed_model",
        "routed_models",
        "used_models",
        "correct_routing",
        "critical",
        "escalated",
        "latency_s",
        "graph_latency_s",
        "cost_usd",
        "saved_vs_gpt5_usd",
        "knn_top_score",
        "flow_nodes",
        "can_self_answer",
        "failure_type",
        "category",
        "baseline_cost",
        "saved_vs_baseline",
        "success",
        "quality_score",
        "quality_cost_usd",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for row in results:
            serializable = dict(row)
            serializable["routed_models"] = "|".join(serializable["routed_models"])
            serializable["used_models"] = "|".join(serializable["used_models"])
            serializable["flow_nodes"] = "|".join(serializable["flow_nodes"])
            serializable.pop("quality_dimensions", None)
            writer.writerow({k: serializable[k] for k in csv_headers if k in serializable})
    with open(latest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for row in results:
            serializable = dict(row)
            serializable["routed_models"] = "|".join(serializable["routed_models"])
            serializable["used_models"] = "|".join(serializable["used_models"])
            serializable["flow_nodes"] = "|".join(serializable["flow_nodes"])
            serializable.pop("quality_dimensions", None)
            writer.writerow({k: serializable[k] for k in csv_headers if k in serializable})

    return summary, results, json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end benchmark with latency/model-flow reporting.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N queries (0 = all).")
    parser.add_argument(
        "--query-timeout-s",
        type=float,
        default=90.0,
        help="Hard timeout per query in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = args.limit if args.limit > 0 else None
    summary, _, json_path, csv_path = asyncio.run(
        run_e2e_benchmark(limit=limit, query_timeout_s=args.query_timeout_s)
    )
    print("\nE2E BENCHMARK SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nJSON report: {json_path}")
    print(f"CSV report:  {csv_path}")
    print("Latest JSON: eval/e2e_benchmark_latest.json")
    print("Latest CSV:  eval/e2e_benchmark_latest.csv")


if __name__ == "__main__":
    main()
