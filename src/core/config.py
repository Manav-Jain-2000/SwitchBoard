import os

# === Model Constants (Build Spec: 7 Models) ===

# Classifier (Cerebras — ultra-low latency)
MODEL_CLASSIFIER = "cerebras/llama3.1-8b"

# Worker Models (selected by KNN router)
MODEL_LLAMA_GROQ = "groq/llama-3.1-8b-instant"                            # Simple Q&A
MODEL_KIMI_K2 = "groq/meta-llama/llama-4-scout-17b-16e-instruct"          # Code low-medium (Llama 4 Scout via Groq)
MODEL_GPT_OSS = "groq/openai/gpt-oss-120b"                                # General medium (gpt-oss-120b via Groq)
MODEL_QWEN_235B = "cerebras/qwen-3-235b-a22b-instruct-2507" # Research, complex reasoning (updated to Cerebras)
MODEL_GPT4O = "openrouter/openai/gpt-4o"                 # Critical Q&A, factual research (via OpenRouter)
MODEL_GEMINI_FLASH = "gemini/gemini-2.5-flash"           # Math, aggregator, judge
MODEL_OPUS = "openrouter/anthropic/claude-opus-4.6"      # Critical code only

# Special-purpose models
JUDGE_MODEL = MODEL_GEMINI_FLASH
AGGREGATOR_MODEL = MODEL_GEMINI_FLASH
MODEL_EMBED = "fireworks_ai/nomic-ai/nomic-embed-text-v1.5"

# Thresholds
JUDGE_THRESHOLD = 7.0
MAX_ESCALATIONS = 1
KNN_K_VALUE = 5  # top-5 KNN vote

# Per-category judge thresholds — lower threshold = fewer escalations
JUDGE_THRESHOLDS = {
    "critical": 7.0,        # medical/legal — keep strict
    "critical_code": 6.0,   # code quality — more lenient to avoid costly Opus escalation
    "default": 7.0,
}

# Per-category escalation models — use cheaper models where possible
ESCALATION_MODELS = {
    "critical": MODEL_OPUS,       # medical/legal needs the best
    "critical_code": MODEL_GPT4O, # code can use GPT-4o (10x cheaper than Opus)
    "default": MODEL_GPT4O,       # default escalation to GPT-4o instead of Opus
}

# GPT-5 baseline cost per query (for savings calculation).
# Override via env when you have your own measured baseline for your workload.
GPT5_BASELINE_COST = float(os.getenv("GPT5_BASELINE_COST", "0.012"))

# Costs per 1M tokens (USD) — approximate
MODEL_COSTS = {
    MODEL_CLASSIFIER: {"input": 0.10, "output": 0.10},
    MODEL_LLAMA_GROQ: {"input": 0.05, "output": 0.08},
    MODEL_KIMI_K2: {"input": 0.20, "output": 0.20},
    MODEL_GPT_OSS: {"input": 0.50, "output": 0.50},
    MODEL_QWEN_235B: {"input": 0.40, "output": 0.40},
    MODEL_GPT4O: {"input": 2.50, "output": 10.00},
    MODEL_GEMINI_FLASH: {"input": 0.075, "output": 0.30},
    MODEL_OPUS: {"input": 15.00, "output": 75.00},
}

# Per-category GPT-5 baseline costs (USD per query, estimated)
# These reflect what it would cost to send each query type to GPT-5 directly
CATEGORY_BASELINES = {
    "simple_qa": 0.002,      # Short response, low tokens
    "code": 0.008,           # Medium response, moderate tokens
    "general": 0.006,        # Medium response
    "research": 0.015,       # Long response, high tokens
    "critical": 0.012,       # Medium-high response
    "math": 0.004,           # Short response
    "critical_code": 0.020,  # Long code response
}
