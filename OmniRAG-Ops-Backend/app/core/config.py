from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    COLLECTION_NAME: str = "omnirarg_docs"

    GEMINI_MODEL: str = "gemini-2.5-flash"

    EMBEDDING_MODEL_ID: str = "models/gemini-embedding-2"
    EMBEDDING_OUTPUT_DIMENSIONALITY: int = 1536

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    DATABASE_URL: str = "postgresql+asyncpg://omnirg:omnirg@db:5432/omnirg"
    RECORD_MANAGER_NAMESPACE: str = "omnirag/upsert_records"

    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
