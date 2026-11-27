"""Validated runtime settings for Medha's application and brain adapter."""
from __future__ import annotations

import os
from dataclasses import dataclass

from backend.privacy import NOTICE_VERSION


def _integer(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _floating(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


@dataclass(frozen=True)
class Settings:
    brain_provider: str = "stub"
    brain_base_url: str = "http://127.0.0.1:11434/v1"
    brain_model: str | None = None
    brain_api_key: str | None = None
    brain_timeout_seconds: float = 45.0
    brain_retries: int = 2
    max_concurrent_generations: int = 1
    max_queued_generations: int = 4
    generation_queue_timeout_seconds: float = 45.0
    max_output_tokens: int = 1024
    history_token_budget: int = 8192
    search_synthesis_enabled: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    experimental_checkpoint: str = "medha_local_llm/checkpoints/medha.pt"
    database_path: str = "instance/medha.sqlite3"
    max_users: int = 100
    signups_enabled: bool = True
    maintenance: bool = False
    read_only: bool = False
    cookie_secure: bool = True
    session_ttl_seconds: int = 604800
    bootstrap_invite_code: str | None = None
    max_request_bytes: int = 65536
    max_conversation_messages: int = 200
    chat_ip_rate_per_minute: int = 120
    chat_user_rate_per_minute: int = 40
    log_retention_days: int = 30
    consent_notice_version: str = NOTICE_VERSION
    grievance_email: str | None = None
    sentry_dsn: str | None = None
    upload_storage_dir: str = "instance/uploads"
    max_upload_bytes: int = 5 * 1024 * 1024
    user_upload_quota_bytes: int = 25 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        model = os.getenv("MEDHA_BRAIN_MODEL", "").strip() or None
        api_key = os.getenv("MEDHA_BRAIN_API_KEY", "").strip() or None
        return cls(
            brain_provider=os.getenv("MEDHA_BRAIN_PROVIDER", "stub").strip().lower(),
            brain_base_url=os.getenv("MEDHA_BRAIN_BASE_URL", "http://127.0.0.1:11434/v1").strip(),
            brain_model=model,
            brain_api_key=api_key,
            brain_timeout_seconds=_floating("MEDHA_BRAIN_TIMEOUT_SECONDS", 45.0),
            brain_retries=_integer("MEDHA_BRAIN_RETRIES", 2),
            max_concurrent_generations=_integer("MEDHA_MAX_CONCURRENT_GENERATIONS", 1),
            max_queued_generations=_integer("MEDHA_MAX_QUEUED_GENERATIONS", 4),
            generation_queue_timeout_seconds=_floating("MEDHA_GENERATION_QUEUE_TIMEOUT_SECONDS", 45.0),
            max_output_tokens=_integer("MEDHA_MAX_OUTPUT_TOKENS", 1024),
            history_token_budget=_integer("MEDHA_HISTORY_TOKEN_BUDGET", 8192),
            search_synthesis_enabled=_boolean("MEDHA_BRAIN_SEARCH_SYNTHESIS", False),
            temperature=_floating("MEDHA_TEMPERATURE", 0.7),
            top_p=_floating("MEDHA_TOP_P", 0.9),
            experimental_checkpoint=os.getenv(
                "MEDHA_EXPERIMENTAL_CHECKPOINT", "medha_local_llm/checkpoints/medha.pt"
            ),
            database_path=os.getenv("MEDHA_DATABASE_PATH", "instance/medha.sqlite3"),
            max_users=_integer("MEDHA_MAX_USERS", 100),
            signups_enabled=_boolean("MEDHA_SIGNUPS_ENABLED", True),
            maintenance=_boolean("MEDHA_MAINTENANCE", False),
            read_only=_boolean("MEDHA_READ_ONLY", False),
            cookie_secure=_boolean("MEDHA_COOKIE_SECURE", True),
            session_ttl_seconds=_integer("MEDHA_SESSION_TTL_SECONDS", 604800),
            bootstrap_invite_code=os.getenv("MEDHA_BOOTSTRAP_INVITE_CODE") or None,
            max_request_bytes=_integer("MEDHA_MAX_REQUEST_BYTES", 65536),
            max_conversation_messages=_integer("MEDHA_MAX_CONVERSATION_MESSAGES", 200),
            chat_ip_rate_per_minute=_integer("MEDHA_CHAT_IP_RATE_PER_MINUTE", 120),
            chat_user_rate_per_minute=_integer("MEDHA_CHAT_USER_RATE_PER_MINUTE", 40),
            log_retention_days=_integer("MEDHA_LOG_RETENTION_DAYS", 30),
            consent_notice_version=os.getenv("MEDHA_CONSENT_NOTICE_VERSION", NOTICE_VERSION).strip(),
            grievance_email=os.getenv("GRIEVANCE_EMAIL", "").strip() or None,
            sentry_dsn=os.getenv("SENTRY_DSN", "").strip() or None,
            upload_storage_dir=os.getenv("MEDHA_UPLOAD_DIR", "instance/uploads").strip(),
            max_upload_bytes=_integer("MEDHA_MAX_UPLOAD_BYTES", 5 * 1024 * 1024),
            user_upload_quota_bytes=_integer("MEDHA_USER_UPLOAD_QUOTA_BYTES", 25 * 1024 * 1024),
        )

    def validate(self) -> None:
        if self.brain_provider not in {"stub", "medha", "openai_compat"}:
            raise ValueError("MEDHA_BRAIN_PROVIDER must be stub, medha, or openai_compat.")
        if self.brain_provider == "openai_compat" and not self.brain_model:
            raise ValueError("MEDHA_BRAIN_MODEL is required for the openai_compat brain.")
        if self.brain_provider == "openai_compat" and not self.brain_base_url.startswith(("http://", "https://")):
            raise ValueError("MEDHA_BRAIN_BASE_URL must use http or https.")
        if self.brain_timeout_seconds <= 0 or self.generation_queue_timeout_seconds <= 0:
            raise ValueError("Brain and queue timeouts must be positive.")
        if self.brain_retries < 0 or self.max_concurrent_generations < 1:
            raise ValueError("Retries cannot be negative and concurrency must be positive.")
        if self.max_queued_generations < 0 or self.max_output_tokens < 1 or self.history_token_budget < 1:
            raise ValueError("Queue, output, and context limits must be non-negative and useful.")
        if not 0 < self.top_p <= 1 or self.temperature <= 0:
            raise ValueError("Sampling values must be positive and top_p cannot exceed one.")
        if self.max_users < 1 or self.session_ttl_seconds < 60:
            raise ValueError("User capacity must be positive and sessions must last at least one minute.")
        if min(self.max_request_bytes, self.chat_ip_rate_per_minute, self.chat_user_rate_per_minute) < 1:
            raise ValueError("Request and rate limits must be positive.")
        if self.max_conversation_messages < 2:
            raise ValueError("Conversation limit must allow at least one user and assistant message.")
        if not 1 <= self.log_retention_days <= 3650:
            raise ValueError("MEDHA_LOG_RETENTION_DAYS must be between 1 and 3650.")
        if not self.consent_notice_version or len(self.consent_notice_version) > 64:
            raise ValueError("MEDHA_CONSENT_NOTICE_VERSION must be a non-empty version label of at most 64 characters.")
        if self.sentry_dsn and not self.sentry_dsn.startswith("https://"):
            raise ValueError("SENTRY_DSN must be an https URL when configured.")
        if not self.database_path.strip():
            raise ValueError("MEDHA_DATABASE_PATH cannot be empty.")
        if not self.upload_storage_dir.strip():
            raise ValueError("MEDHA_UPLOAD_DIR cannot be empty.")
        if self.max_upload_bytes < 1 or self.user_upload_quota_bytes < self.max_upload_bytes:
            raise ValueError("Upload limits must be positive and the quota must cover one maximum-sized file.")
