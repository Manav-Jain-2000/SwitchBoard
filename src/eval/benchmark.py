import asyncio
import time
import json
import uuid
from langgraph.types import Command
from core.graph import switchboard_graph
from core.config import (
    GPT5_BASELINE_COST,
    MODEL_LLAMA_GROQ,
    MODEL_KIMI_K2,
    MODEL_GPT_OSS,
    MODEL_QWEN_235B,
    MODEL_GPT4O,
    MODEL_GEMINI_FLASH,
    MODEL_OPUS,
)

# 60 realistic queries with expected model routing (approx 8-9 per category)
QUERIES = [
    # Simple Q&A — expect Llama Groq
    {"q": "Hello, how are you?", "expected": MODEL_LLAMA_GROQ, "category": "simple_qa"},
    {"q": "What is the capital of Japan?", "expected": MODEL_LLAMA_GROQ, "category": "simple_qa"},
    {"q": "Who invented the telephone?", "expected": MODEL_LLAMA_GROQ, "category": "simple_qa"},
    {"q": "What does DNA stand for?", "expected": MODEL_LLAMA_GROQ, "category": "simple_qa"},
    {"q": "How many days are in a leap year?", "expected": MODEL_LLAMA_GROQ, "category": "simple_qa"},
    {"q": "What is the largest ocean on Earth?", "expected": MODEL_LLAMA_GROQ, "category": "simple_qa"},
    {"q": "Who painted the Mona Lisa?", "expected": MODEL_LLAMA_GROQ, "category": "simple_qa"},
    {"q": "What is the speed of light?", "expected": MODEL_LLAMA_GROQ, "category": "simple_qa"},
    {"q": "Define entropy in one sentence.", "expected": MODEL_LLAMA_GROQ, "category": "simple_qa"},

    # Code — expect Kimi K2
    {"q": "Write a Python function to merge two sorted arrays.", "expected": MODEL_KIMI_K2, "category": "code"},
    {"q": "Implement a stack using linked lists in Java.", "expected": MODEL_KIMI_K2, "category": "code"},
    {"q": "Debug this TypeScript function that returns undefined.", "expected": MODEL_KIMI_K2, "category": "code"},
    {"q": "Write unit tests for a REST API endpoint using pytest.", "expected": MODEL_KIMI_K2, "category": "code"},
    {"q": "Create a React hook for debounced search.", "expected": MODEL_KIMI_K2, "category": "code"},
    {"q": "Implement the observer pattern in Python.", "expected": MODEL_KIMI_K2, "category": "code"},
    {"q": "Write a SQL query to find the second highest salary.", "expected": MODEL_KIMI_K2, "category": "code"},
    {"q": "Convert this Python 2 code to Python 3.", "expected": MODEL_KIMI_K2, "category": "code"},
    {"q": "Write a basic web scraper in Python using BeautifulSoup.", "expected": MODEL_KIMI_K2, "category": "code"},

    # General — expect GPT OSS 120B
    {"q": "Compare the pros and cons of electric vs hydrogen cars.", "expected": MODEL_GPT_OSS, "category": "general"},
    {"q": "Explain how blockchain consensus mechanisms work.", "expected": MODEL_GPT_OSS, "category": "general"},
    {"q": "What are the key differences between agile and waterfall?", "expected": MODEL_GPT_OSS, "category": "general"},
    {"q": "Explain the model-view-controller architecture pattern.", "expected": MODEL_GPT_OSS, "category": "general"},
    {"q": "Describe how DNS resolution works step by step.", "expected": MODEL_GPT_OSS, "category": "general"},
    {"q": "Compare Docker and virtual machines.", "expected": MODEL_GPT_OSS, "category": "general"},
    {"q": "What are the trade-offs of eventual consistency?", "expected": MODEL_GPT_OSS, "category": "general"},
    {"q": "How does HTTPS encryption work?", "expected": MODEL_GPT_OSS, "category": "general"},
    {"q": "Explain the difference between threads and processes.", "expected": MODEL_GPT_OSS, "category": "general"},

    # Research — expect Qwen 235B
    {"q": "Analyze the economic impact of AI on the labor market in 2025.", "expected": MODEL_QWEN_235B, "category": "research"},
    {"q": "Write a comprehensive overview of mRNA vaccine technology.", "expected": MODEL_QWEN_235B, "category": "research"},
    {"q": "Explain the geopolitical implications of semiconductor supply chains.", "expected": MODEL_QWEN_235B, "category": "research"},
    {"q": "Provide a deep analysis of central bank digital currencies.", "expected": MODEL_QWEN_235B, "category": "research"},
    {"q": "Summarize the evolution of machine learning from 1950 to present.", "expected": MODEL_QWEN_235B, "category": "research"},
    {"q": "Analyze the sociological effects of social media on Gen Z.", "expected": MODEL_QWEN_235B, "category": "research"},
    {"q": "Write a literature review on attention mechanisms in NLP.", "expected": MODEL_QWEN_235B, "category": "research"},
    {"q": "Compare the education systems of Finland, Japan, and the US.", "expected": MODEL_QWEN_235B, "category": "research"},

    # Critical — expect GPT-4o
    {"q": "Is it safe to combine metformin and alcohol?", "expected": MODEL_GPT4O, "category": "critical"},
    {"q": "What are my rights if wrongfully terminated in Texas?", "expected": MODEL_GPT4O, "category": "critical"},
    {"q": "Explain the tax implications of selling inherited property.", "expected": MODEL_GPT4O, "category": "critical"},
    {"q": "What are the warning signs of a stroke?", "expected": MODEL_GPT4O, "category": "critical"},
    {"q": "Review this non-compete clause for enforceability issues.", "expected": MODEL_GPT4O, "category": "critical"},
    {"q": "What are the fiduciary duties of a board of directors?", "expected": MODEL_GPT4O, "category": "critical"},
    {"q": "Explain HIPAA compliance requirements for a web app.", "expected": MODEL_GPT4O, "category": "critical"},
    {"q": "What are the side effects of long-term statin use?", "expected": MODEL_GPT4O, "category": "critical"},

    # Math — expect Gemini Flash
    {"q": "Solve for x: 5x^2 - 3x + 2 = 0.", "expected": MODEL_GEMINI_FLASH, "category": "math"},
    {"q": "What is the derivative of ln(x^2 + 1)?", "expected": MODEL_GEMINI_FLASH, "category": "math"},
    {"q": "Calculate compound interest: $5000 at 4% for 10 years.", "expected": MODEL_GEMINI_FLASH, "category": "math"},
    {"q": "Convert 250 kilometers per hour to miles per hour.", "expected": MODEL_GEMINI_FLASH, "category": "math"},
    {"q": "Find the eigenvalues of matrix [[2,1],[1,2]].", "expected": MODEL_GEMINI_FLASH, "category": "math"},
    {"q": "Integrate x*e^x dx.", "expected": MODEL_GEMINI_FLASH, "category": "math"},
    {"q": "Compute the probability of rolling sum 7 with two dice.", "expected": MODEL_GEMINI_FLASH, "category": "math"},

    # Critical code — expect Opus
    {"q": "Write production JWT auth middleware with refresh tokens for Express.", "expected": MODEL_OPUS, "category": "critical_code"},
    {"q": "Design a scalable event-driven payment processing system.", "expected": MODEL_OPUS, "category": "critical_code"},
    {"q": "Perform a security audit on this REST API implementation.", "expected": MODEL_OPUS, "category": "critical_code"},
    {"q": "Implement distributed locking with Redis and proper failover.", "expected": MODEL_OPUS, "category": "critical_code"},
    {"q": "Write a complete CI/CD pipeline for Kubernetes deployment.", "expected": MODEL_OPUS, "category": "critical_code"},
    {"q": "Design the database schema for encrypted real-time messaging.", "expected": MODEL_OPUS, "category": "critical_code"},
]


async def run_benchmark():
    print(f"Running {len(QUERIES)}-query benchmark...")

    results = []
    total_cost = 0.0
    critical_queries = 0
    escalations = 0
    latency_total = 0.0
    correct_routing = 0
    knn_confidence_total = 0.0

    for idx, item in enumerate(QUERIES):
        query = item["q"]
        expected = item["expected"]

        session_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}

        start_time = time.time()
        initial_state = {"query": query}

        try:
            final_state = None
            async for event in switchboard_graph.astream(initial_state, config=config, stream_mode="values"):
                final_state = event

            # If HITL was triggered, resume with default answer
            state_snapshot = switchboard_graph.get_state(config)
            if state_snapshot.next and "hitl" in state_snapshot.next:
                async for event in switchboard_graph.astream(
                    Command(resume="please proceed with best guess"), config=config, stream_mode="values"
                ):
                    final_state = event

            end_time = time.time()
            latency = end_time - start_time
            latency_total += latency

            final_memory = switchboard_graph.get_state(config).values

            selected_models = final_memory.get("selected_models", [])
            routed_model = selected_models[0] if selected_models else "unknown"
            is_critical = final_memory.get("is_critical", False)
            escalated = final_memory.get("escalation_count", 0) > 0
            query_cost = final_memory.get("total_cost", 0.0)
            knn_scores = final_memory.get("knn_scores", {})
            top_knn = max(knn_scores.values()) if knn_scores else 0.0

            if routed_model == expected:
                correct_routing += 1

            if is_critical:
                critical_queries += 1

            if escalated:
                escalations += 1

            total_cost += query_cost
            knn_confidence_total += top_knn

            results.append({
                "id": idx,
                "query": query,
                "expected": expected,
                "routed": routed_model,
                "correct": routed_model == expected,
                "critical": is_critical,
                "escalated": escalated,
                "latency_s": round(latency, 2),
                "cost_usd": round(query_cost, 6),
                "knn_top_score": round(top_knn, 4),
            })

            status = "OK" if routed_model == expected else "❌"
            print(f"[{idx+1}/{len(QUERIES)}] {status} routed={routed_model} expected={expected} latency={latency:.1f}s")

        except Exception as e:
            print(f"[{idx+1}/{len(QUERIES)}] ❌ Failed: {e}")

    # Summary
    n = len(QUERIES)
    avg_cost = total_cost / n if n > 0 else 0
    baseline_cost = GPT5_BASELINE_COST * n
    saved = baseline_cost - total_cost
    saved_pct = (saved / baseline_cost * 100) if baseline_cost > 0 else 0

    summary = {
        "total_queries": n,
        "routing_accuracy": f"{(correct_routing / n) * 100:.1f}%",
        "avg_cost_per_query": f"${avg_cost:.6f}",
        "total_cost": f"${total_cost:.6f}",
        "vs_gpt5_baseline": f"${baseline_cost:.4f}",
        "cost_saved": f"${saved:.4f} ({saved_pct:.1f}%)",
        "judge_triggered": f"{critical_queries}/{n}",
        "escalations": f"{escalations}/{n}",
        "avg_knn_confidence": f"{knn_confidence_total / n:.4f}" if n > 0 else "N/A",
        "avg_latency": f"{latency_total / n:.2f}s" if n > 0 else "N/A",
    }

    print(f"\n{'='*50}")
    print("BENCHMARK SUMMARY")
    print("="*50)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    final_output = {"summary": summary, "results": results}

    with open("eval/results.json", "w") as f:
        json.dump(final_output, f, indent=4)

    print("\nResults saved to eval/results.json")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
