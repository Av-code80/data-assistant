"""Application settings.

Everything that differs between dev and prod comes from the environment
(variables or a local .env file that is never committed). Values are typed and
validated at startup, so a missing API key stops the app immediately.
"""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # SecretStr keeps the key out of logs and tracebacks. Read from ANTHROPIC_API_KEY.
    anthropic_api_key: SecretStr

    llm_model: str = "claude-opus-5-5"
    llm_max_tokens: int = 4000
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2  # the SDK retries 429, 5xx and network errors on its own

    # USD per million tokens, used to report the cost of every call.
    input_price_per_million: float = 4.00
    output_price_per_million: float = 20.00

    # Upper bound on the result rows sent to the LLM: bounds cost and latency.
    max_answer_rows: int = 200

    # Dates covered by the database. Periods outside are clamped or rejected by the
    # server, never left to the LLM. In production this is read from the warehouse.
    data_available_from: date = date(2024, 1, 1)
    data_available_until: date = date(2026, 10, 24)

    # Source tables, and the view layer built from them: the only file the SQL agent reads.
    source_db_path: Path = PROJECT_DIR / "data" / "shop.duckdb"
    analytics_db_path: Path = PROJECT_DIR / "data" / "analytics.duckdb"
    sql_max_rows: int = 200       # rows the SQL tool returns to the agent
    sql_timeout_s: float = 5.0    # a query running longer is interrupted

    # SQL agent: a hard cap on LLM calls per question bounds cost and latency.
    agent_max_steps: int = 6
    agent_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    agent_preview_rows: int = 20  # result rows the agent sees after each query

    # Retrieval (RAG) over the business knowledge base.
    knowledge_dir: Path = PROJECT_DIR / "knowledge"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_cache_dir: Path = PROJECT_DIR / ".cache" / "models"
    retrieval_top_k: int = 3
    retrieval_min_vector_score: float = 0.0

    cors_origins: list[str] = ["http://localhost:5173"]  # Vite dev server


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them."""
    return Settings()
