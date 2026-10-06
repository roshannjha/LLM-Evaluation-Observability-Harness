from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DIR = DATA_DIR / "chroma"
GOLDEN_DIR = DATA_DIR / "golden"
REPORTS_DIR = REPO_ROOT / "reports"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    embedding_backend: Literal["local", "openai"] = "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"

    llm_backend: Literal["openai", "grok", "gemini"] = "openai"
    generation_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"

    openai_api_key: str | None = None
    xai_api_key: str | None = None
    xai_base_url: str = "https://api.x.ai/v1"
    google_api_key: str | None = None

    collection_name: str = "nike_filings"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    LANGFUSE_BASE_URL : str = "https://us.cloud.langfuse.com"

    faithfulness_min: float = Field(0.90, ge=0.0, le=1.0)
    toxicity_max: float = Field(0.05, ge=0.0, le=1.0)

    @property
    def chroma_dir(self) -> Path:
        return CHROMA_DIR

    @property
    def corpus_dir(self) -> Path:
        return CORPUS_DIR


settings = Settings()


if __name__ == "__main__":
    import json

    print(json.dumps(settings.model_dump(), indent=2, default=str))
    print(f"\nRepo root : {REPO_ROOT}")
    print(f"Corpus    : {CORPUS_DIR}  (exists={CORPUS_DIR.exists()})")
    print(f"Chroma    : {CHROMA_DIR}")
