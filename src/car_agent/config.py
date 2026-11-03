"""Runtime settings read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path
    groq_model: str = "openai/gpt-oss-120b"
    reasoning_effort: str = "low"        # gpt-oss models: low / medium / high
    temperature: float = 0.3             # low temperature -> more reliable tool calls
    max_completion_tokens: int = 4_000   # reasoning models spend part of it on hidden reasoning
    max_tool_rounds: int = 5             # tool-calling rounds per user message
    max_history_turns: int = 6           # previous user/assistant exchanges kept as context
    max_results: int = 10                # upper limit of listings returned by one tool call
    max_tool_result_chars: int = 6_000   # tool output sent back to the LLM is truncated beyond this


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts/car_price_model"))
    if not artifacts_dir.is_absolute():
        artifacts_dir = PROJECT_ROOT / artifacts_dir
    return Settings(
        artifacts_dir=artifacts_dir,
        groq_model=os.getenv("GROQ_MODEL", Settings.groq_model),
        reasoning_effort=os.getenv("REASONING_EFFORT", Settings.reasoning_effort),
    )
