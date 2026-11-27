# MedhaAI launch architecture

## Request path

The browser loads the existing Medha OS client from `backend/`. Same-origin JSON endpoints handle account and conversation operations. `POST /api/chat` persists the user message, invokes the configured brain through a bounded queue, and emits Server-Sent Events (`token`, `sources`, `done`, or `error`). Search will add sources in WP5; before that work lands, the API reports an empty source list and the UI does not pretend to have searched.

## Brain boundary

`backend/medha_brain.py` owns the `generate_stream(messages, params)` interface. The production selection is a local OpenAI-compatible endpoint such as Ollama. `MEDHA_BRAIN_MODEL` is supplied by the operator and has no application default. The 2.7M-parameter from-scratch model is an experimental backend only. The deterministic stub and an in-process fake HTTP server keep tests offline.

The brain receives a short, versioned system prompt and bounded history. The request's newest user turn and system prompt are kept while older turns are trimmed. The adapter caps completion length, uses finite network timeouts, and retries only before it has yielded output.

## Persistence and trust boundaries

SQLite runs in WAL mode with numbered migrations. Passwords use scrypt; sessions store token hashes rather than bearer tokens. Mutations require CSRF validation and ownership checks. Uploaded files and web pages are untrusted data; they must never override the system prompt. The UI treats API content as text and builds rendered nodes using DOM APIs.

Logs contain request metadata and counts, never prompt bodies, credentials, session tokens, or file contents. Database, uploads, model weights, and environment files stay outside Git.

## Deployment shape

One FastAPI worker serves the UI and API. The local model server is an optional `local-brain` Compose profile, allowing the owner to install and benchmark the selected weights separately. SQLite and uploads use persistent volumes. A reverse proxy terminates HTTPS so Secure cookies can be used. Rate limits and generation queue counters are process-local; launch therefore uses one app worker.
