"""Streaming adapters for Medha's replaceable assistant brain."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Condition
from typing import Protocol

from backend.config import Settings

Message = Mapping[str, str]
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "system.md"


class BrainError(RuntimeError):
    """A safe-to-show failure from a configured brain provider."""


class GenerationCapacityError(RuntimeError):
    """Raised when the bounded generation queue is full or times out."""


@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


class Brain(Protocol):
    def generate_stream(self, messages: Sequence[Message], params: GenerationParams) -> Iterator[str]: ...

    def health(self) -> bool: ...

    def warmup(self) -> None: ...


def _token_estimate(message: Message) -> int:
    # Four UTF-8 characters per token is a conservative, dependency-free budget estimate.
    return max(1, (len(message.get("content", "")) + 3) // 4) + 4


def trim_history(messages: Sequence[Message], token_budget: int) -> list[dict[str, str]]:
    """Keep the system prompt and newest user request while trimming old turns."""
    if token_budget < 1:
        raise ValueError("token_budget must be positive.")
    normalized = [dict(message) for message in messages if message.get("role") in {"system", "user", "assistant"}]
    system = next((message for message in normalized if message["role"] == "system"), None)
    turns = [message for message in normalized if message["role"] != "system"]
    last_user_index = next((i for i in range(len(turns) - 1, -1, -1) if turns[i]["role"] == "user"), None)
    required: set[int] = set()
    used = 0
    if system is not None:
        used += _token_estimate(system)
    if last_user_index is not None:
        required.add(last_user_index)
        used += _token_estimate(turns[last_user_index])

    # If the required prompt is already over budget, it is still safer to keep it intact.
    for index in range(len(turns) - 1, -1, -1):
        if index in required:
            continue
        cost = _token_estimate(turns[index])
        if used + cost <= token_budget:
            required.add(index)
            used += cost
    kept = [turn for index, turn in enumerate(turns) if index in required]
    return ([system] if system is not None else []) + kept


def system_prompt() -> str:
    """Read the short, versioned prompt and inject the server's current date."""
    prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    today = datetime.now(timezone.utc).date().isoformat()
    return prompt.replace("{{CURRENT_DATE}}", today)


def prepare_messages(messages: Sequence[Message], token_budget: int) -> list[dict[str, str]]:
    """Prevent client-supplied system messages from replacing Medha's instructions."""
    conversation = [dict(message) for message in messages if message.get("role") in {"user", "assistant"}]
    return trim_history([{"role": "system", "content": system_prompt()}, *conversation], token_budget)


class StubBrain:
    """A deterministic backend used for tests and local UI development."""

    def __init__(self, reply: str = "This is a deterministic Medha test response.") -> None:
        self.reply = reply

    def generate_stream(self, messages: Sequence[Message], params: GenerationParams) -> Iterator[str]:
        del messages, params
        for start in range(0, len(self.reply), 8):
            yield self.reply[start:start + 8]

    def health(self) -> bool:
        return True

    def warmup(self) -> None:
        return None


class OpenAICompatBrain:
    """Stream chat completions from a local OpenAI-compatible service."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 45.0,
        retries: int = 2,
        max_output_tokens: int = 1024,
        opener: Callable[..., object] = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model:
            raise ValueError("A model name is required for the OpenAI-compatible brain.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.max_output_tokens = max_output_tokens
        self._opener = opener
        self._sleep = sleeper

    @property
    def _completion_url(self) -> str:
        return self.base_url + "/chat/completions"

    def _request(self, messages: Sequence[Message], params: GenerationParams) -> urllib.request.Request:
        body = json.dumps({
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": min(params.max_tokens, self.max_output_tokens),
        }).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        return urllib.request.Request(self._completion_url, data=body, headers=headers, method="POST")

    def _stream_once(self, request: urllib.request.Request) -> Iterator[str]:
        try:
            response = self._opener(request, timeout=self.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BrainError("The configured brain could not be reached.") from exc
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    event = json.loads(data)
                    choices = event.get("choices", [])
                    content = choices[0].get("delta", {}).get("content") if choices else None
                except (json.JSONDecodeError, AttributeError, IndexError, TypeError) as exc:
                    raise BrainError("The configured brain returned an invalid stream event.") from exc
                if isinstance(content, str) and content:
                    yield content
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BrainError("The configured brain stream ended unexpectedly.") from exc
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    def generate_stream(self, messages: Sequence[Message], params: GenerationParams) -> Iterator[str]:
        request = self._request(messages, params)
        for attempt in range(self.retries + 1):
            iterator = self._stream_once(request)
            yielded = False
            try:
                while True:
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        return
                    yielded = True
                    yield chunk
            except BrainError:
                # Retrying after a visible token would duplicate answer text in the client.
                if yielded or attempt >= self.retries:
                    raise
                self._sleep(0.25 * (2 ** attempt))

    def _request_json(self, path: str) -> bool:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.base_url + path, headers=headers, method="GET")
        try:
            response = self._opener(request, timeout=self.timeout)
            response.close()
            return True
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def health(self) -> bool:
        return self._request_json("/models")

    def warmup(self) -> None:
        warmup_prompt = [{"role": "user", "content": "Respond with OK."}]
        try:
            next(self.generate_stream(warmup_prompt, GenerationParams(max_tokens=1)))
        except StopIteration:
            # Some compatible servers complete a 1-token request without emitting text.
            return


class MedhaExperimentalBrain:
    """Adapter for the from-scratch toy model; never selected by default."""

    def __init__(self, checkpoint: str, max_output_tokens: int = 128) -> None:
        self.checkpoint = Path(checkpoint)
        self.max_output_tokens = max_output_tokens
        self._loaded: tuple[object, object, object] | None = None

    def _load(self) -> tuple[object, object, object]:
        if self._loaded is None:
            import torch
            from medha_local_llm.model import load_checkpoint

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, tokenizer = load_checkpoint(self.checkpoint, device)
            self._loaded = model, tokenizer, device
        return self._loaded

    def generate_stream(self, messages: Sequence[Message], params: GenerationParams) -> Iterator[str]:
        import torch
        from medha_local_llm.tokenizer import ByteBPETokenizer

        model, tokenizer, device = self._load()
        assert isinstance(tokenizer, ByteBPETokenizer)
        token_ids = tokenizer.encode_chat(messages, add_assistant_prompt=True)
        context_size = model.config.block_size
        # Reserve part of the toy model's tiny context for an answer before trimming.
        prompt_limit = max(1, context_size - min(32, max(1, context_size // 4)))
        token_ids = token_ids[-prompt_limit:]
        room = context_size - len(token_ids)
        limit = min(params.max_tokens, self.max_output_tokens, room)
        if limit == 0:
            yield ""
            return
        prompt = torch.tensor([token_ids[-context_size:]], dtype=torch.long, device=device)
        generated = model.generate(
            prompt,
            max_new_tokens=limit,
            temperature=params.temperature,
            top_p=params.top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(generated[0, prompt.shape[1]:].tolist(), skip_special_tokens=True)
        if text:
            yield text

    def health(self) -> bool:
        return self.checkpoint.is_file()

    def warmup(self) -> None:
        self._load()


def build_brain(settings: Settings) -> Brain:
    settings.validate()
    if settings.brain_provider == "stub":
        return StubBrain()
    if settings.brain_provider == "medha":
        return MedhaExperimentalBrain(settings.experimental_checkpoint, settings.max_output_tokens)
    return OpenAICompatBrain(
        base_url=settings.brain_base_url,
        model=settings.brain_model or "",
        api_key=settings.brain_api_key,
        timeout=settings.brain_timeout_seconds,
        retries=settings.brain_retries,
        max_output_tokens=settings.max_output_tokens,
    )


class GenerationGate:
    """Bound active generations and queued requests without an unbounded task backlog."""

    def __init__(self, max_active: int = 1, max_queued: int = 4, queue_timeout: float = 45.0) -> None:
        if max_active < 1 or max_queued < 0 or queue_timeout <= 0:
            raise ValueError("Generation gate limits are invalid.")
        self.max_active = max_active
        self.max_queued = max_queued
        self.queue_timeout = queue_timeout
        self._condition = Condition()
        self._active = 0
        self._waiting = 0

    def acquire(self) -> None:
        deadline = time.monotonic() + self.queue_timeout
        with self._condition:
            if self._active < self.max_active:
                self._active += 1
                return
            if self._waiting >= self.max_queued:
                raise GenerationCapacityError("Generation queue is full.")
            self._waiting += 1
            try:
                while self._active >= self.max_active:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise GenerationCapacityError("Generation queue wait timed out.")
                    self._condition.wait(remaining)
                self._active += 1
            finally:
                self._waiting -= 1

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("Generation gate released without a matching acquire.")
            self._active -= 1
            self._condition.notify()

    @property
    def active(self) -> int:
        with self._condition:
            return self._active

    @property
    def waiting(self) -> int:
        with self._condition:
            return self._waiting
