import os
import sys

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


REQUIRED_KEYS = (
    "FIREWORKS_API_KEY",
    "GROQ_API_KEY",
    "CEREBRAS_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)


def pytest_collection_modifyitems(config, items):
    """Auto-skip live tests when API keys are missing."""
    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if not missing:
        return
    skip_live = pytest.mark.skip(
        reason=f"Live test skipped — missing env keys: {', '.join(missing)}"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: test that calls real provider APIs (requires .env keys)",
    )
