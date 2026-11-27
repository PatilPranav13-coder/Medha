# Launch decisions

## 2026-11-27 — Rebuild the missing model refactor foundation locally

The connected GitHub default branch returned 404 for `medha_local_llm/tokenizer.py`; the local checkout also lacked that file and the model test directory. The user explicitly directed that missing pieces be built. A dependency-free byte-level BPE tokenizer, version-2 checkpoint support, and tests are being reconstructed locally. This is an experimental research path only; it is not a production brain and its weights are not retrained here.

## 2026-11-27 — Keep the production brain configurable

The product decision is a local open-weights model served through an OpenAI-compatible endpoint. The owner chooses the exact model, licence and quantisation after measurement. The app does not set a model tag or download weights automatically.

## 2026-11-27 — Use same-origin APIs and one app worker

The Medha OS page and API are served by FastAPI from one origin. This avoids a broad CORS policy and lets cookie plus CSRF checks protect mutations. SQLite and in-memory rate/generation controls require a single worker at launch.

## 2026-11-27 — Remove the placeholder frontend after parity

The root Medha OS client now uses the production auth and chat APIs, and the ASGI suite verifies that `/` serves it and exercises signup, streamed chat, conversation history, edit/regenerate, feedback, export, and account deletion. The older standalone `frontend/` echo demo was removed after that parity check so there is one chat client to maintain.
