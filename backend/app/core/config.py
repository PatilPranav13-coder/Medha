from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Load the repository-level .env for local development without overriding
# variables already injected by the process or container runtime.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(value.strip() for value in os.getenv(name, default).split(",") if value.strip())


@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("MEDHA_ENV", "development")
    database_url: str = os.getenv("MEDHA_DATABASE_URL", "sqlite:///./medha.db")
    auth_secret: str = os.getenv("MEDHA_AUTH_SECRET", "")
    cors_origins: tuple[str, ...] = _csv("MEDHA_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    ai_provider: str = os.getenv("MEDHA_AI_PROVIDER", "ollama").strip().lower()
    ai_base_url: str = os.getenv("MEDHA_AI_BASE_URL", "").rstrip("/")
    ai_api_key: str = os.getenv("MEDHA_AI_API_KEY", "")
    ai_model: str = os.getenv("MEDHA_MODEL_NAME", "").strip()
    ai_models: tuple[str, ...] = _csv("MEDHA_MODELS")
    embedding_model: str = os.getenv("MEDHA_EMBEDDING_MODEL", "").strip()
    embedding_dimension: int = int(os.getenv("MEDHA_EMBEDDING_DIMENSION", "768"))
    max_upload_bytes: int = int(os.getenv("MEDHA_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    cookie_secure: bool = os.getenv("MEDHA_COOKIE_SECURE", "false").lower() == "true"

    @property
    def configured_models(self) -> tuple[str, ...]:
        names = self.ai_models or ((self.ai_model,) if self.ai_model else ())
        return tuple(dict.fromkeys(name for name in names if name))


settings = Settings()
