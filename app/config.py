from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    embedding_backend: Literal["local", "openai"] = "local"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_base_url: str = "https://api.openai.com/v1"
    data_dir: Path = Path("data/runtime")
    top_k: int = Field(default=5, ge=1, le=10)
    max_retries: int = Field(default=2, ge=0, le=3)
    chunk_size: int = Field(default=1000, ge=200, le=1600)
    chunk_overlap: int = Field(default=150, ge=0)
    max_upload_bytes: int = Field(default=2 * 1024 * 1024, ge=1, le=10 * 1024 * 1024)
    llm_timeout_seconds: float = Field(default=30, gt=0, le=120)
    query_timeout_seconds: float = Field(default=180, gt=0, le=600)
    check_groundedness: bool = False

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self
