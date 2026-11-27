# Launch decisions

## 2026-11-27 — Rebuild the missing model refactor foundation locally

The connected GitHub default branch returned 404 for `medha_local_llm/tokenizer.py`; the local checkout also lacked that file and the model test directory. The user explicitly directed that missing pieces be built. A dependency-free byte-level BPE tokenizer, version-2 checkpoint support, and tests are being reconstructed locally. This is an experimental research path only; it is not a production brain and its weights are not retrained here.

## 2026-11-27 — Keep the production brain configurable

The product decision is a local open-weights model served through an OpenAI-compatible endpoint. The owner chooses the exact model, licence and quantisation after measurement. The app does not set a model tag or download weights automatically.

## 2026-11-27 — Use same-origin APIs and one app worker

The Medha OS page and API are served by FastAPI from one origin. This avoids a broad CORS policy and lets cookie plus CSRF checks protect mutations. SQLite and in-memory rate/generation controls require a single worker at launch.

## 2026-11-27 — Remove the placeholder frontend after parity

The root Medha OS client now uses the production auth and chat APIs, and the ASGI suite verifies that `/` serves it and exercises signup, streamed chat, conversation history, edit/regenerate, feedback, export, and account deletion. The older standalone `frontend/` echo demo was removed after that parity check so there is one chat client to maintain.

## 2026-11-27 — Use a provider chain with an extractive search floor

Brave and SearXNG are used when configured, with Wikipedia as the no-key last fallback. Auto mode searches checkable or current factual questions and skips small talk, assistant identity, and ordinary code-help prompts. Extractive source sentences are always available; model synthesis remains disabled unless the selected OpenAI-compatible brain is explicitly marked capable. Search text is bounded and encoded as untrusted data, and invalid citations fall back to extractive output. Live network behavior remains unverified; all tests use fixture transports.

## 2026-11-27 — Keep capability claims pending a real model run

The 210-task suite and serving benchmark are local measurement tools. A stub run is only harness validation, so the production thresholds and the ChatGPT Nov 2022 comparison remain pending until the owner selects a model and runs the suite on target hardware. The system prompt is not tuned from stub scores because that would not measure production-model behavior.

## 2026-11-27 — Use small single-worker safety controls

Request-size, IP/user token-bucket, conversation-length and generation limits run in the app process. Their in-memory state requires the documented single-worker launch configuration. Crisis and severe-harm checks are deterministic short-circuits, and untrusted search/file text uses one bounded wrapper for both current and future file Q&A.

## 2026-11-27 — Version consent and minimize operational events

Signup requires the current plain-language notice version and stores it in the same SQLite transaction as account creation. Operational events contain metadata only and default to 30-day retention. Legal files are drafts; commencement dates and notification duties remain owner/legal verification items.
