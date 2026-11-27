"""Validated runtime settings for Medha's application and brain adapter."""
from __future__ import annotations

import os
from dataclasses import dataclass


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
    temperature: float = 0.7
    top_p: float = 0.9
    experimental_checkpoint: str = "medha_local_llm/checkpoints/medha.pt"

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
            temperature=_floating("MEDHA_TEMPERATURE", 0.7),
            top_p=_floating("MEDHA_TOP_P", 0.9),
            experimental_checkpoint=os.getenv(
                "MEDHA_EXPERIMENTAL_CHECKPOINT", "medha_local_llm/checkpoints/medha.pt"
            ),
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
