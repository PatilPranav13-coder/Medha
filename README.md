# Medha

Medha is a technical AI workspace for programming, research, architecture, and engineering. It combines a persistent project workspace with a configurable model provider, streaming chat, public-source lookup, and private project documents.

## What works

- Responsive Next.js workspace with dark/light themes, four technical modes, conversation search/history, Markdown, code highlighting, math, and message actions.
- Email/password registration and login with Argon2 password hashes and signed, HTTP-only sessions.
- Persistent, user-scoped conversations and project workspaces.
- Configurable Ollama or OpenAI-compatible chat streaming; Medha reports a setup error when no model is configured.
- Programming tutor behavior and an app-building target selector for Apple (iPhone/iPad/Mac), Android, Windows, or one Flutter codebase across platforms.
- Optional public developer-document search and Crossref paper discovery. Search can be disabled per message.
- PDF/text/code uploads, text extraction, chunking, optional Ollama/OpenAI-compatible embeddings, pgvector retrieval on PostgreSQL, and lexical fallback.
- User-controlled project and personal memories that can be retrieved in relevant conversations.
- Alembic schema migration, health endpoints, Docker Compose, and backend integration tests.

## Architecture

- `frontend/`: Next.js, TypeScript, Tailwind CSS, and React workspace.
- `backend/app/`: FastAPI routes, SQLAlchemy models, authentication, provider adapters, retrieval, and document processing.
- `backend/alembic/`: PostgreSQL/SQLite schema migrations.
- `docs/`: API, architecture, and development notes.
- `medha_local_llm/` and root HTML/Python files: retained legacy experiments; the new workspace does not depend on them.

The browser talks to FastAPI using an HTTP-only session cookie. The API enforces user ownership for projects, documents, memories, and conversations. Uploaded files are parsed as data and never executed. Public search sends the user's message to the selected public search providers only when its toggle is on. Public snippets and uploaded file excerpts are marked as untrusted in the model context.

Medha uses public documentation and academic metadata as live retrieved context; it does not fine-tune or modify model weights. Public-source lookup is limited to MDN, Microsoft Learn, Stack Exchange, and Crossref in this release. The web-source toggle controls whether the prompt is sent to those services.

## Requirements

- Node.js 22.12+ (Node 26 is used in the frontend container).
- Python 3.11+ (Python 3.13 is used in the backend container).
- PostgreSQL 16 with pgvector for deployment, or SQLite for quick local backend development and tests.
- Ollama with a chat model for local inference, or credentials for an OpenAI-compatible endpoint.
- Docker Compose is optional for containerized startup.

## Configure

Copy `.env.example` to `.env`, replace both placeholder secrets, and set a model provider. Use a long random `MEDHA_AUTH_SECRET`; the database password should be URL-safe (letters and numbers are simplest).

Default Compose configuration uses Ollama and Qwen Coder:

```sh
docker compose up --build -d
docker compose exec ollama ollama pull qwen2.5-coder:7b
docker compose exec ollama ollama pull nomic-embed-text
```

Open <http://localhost:3000>. The API is at <http://localhost:8000>; interactive API docs are at <http://localhost:8000/docs>. The model downloads only when those `ollama pull` commands are run. First response latency and hardware needs depend on the model.

For an OpenAI-compatible service, set `MEDHA_AI_PROVIDER=openai-compatible`, point `MEDHA_AI_BASE_URL` at the provider's base URL, set `MEDHA_AI_API_KEY`, and list the available model IDs in `MEDHA_MODELS`. Embeddings are optional; set `MEDHA_EMBEDDING_MODEL` only when that endpoint supports embeddings. Do not commit `.env` or API keys.

## Run locally without Docker

Start PostgreSQL/pgvector and Ollama separately, then edit `.env` to use host addresses (`localhost` instead of Compose service names `db` and `ollama`). For SQLite development, unset `MEDHA_DATABASE_URL`; the API defaults to `./medha.db`.

Backend (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements-dev.txt
$env:PYTHONPATH = "backend"
python -m alembic -c backend/alembic.ini upgrade head
python -m uvicorn app.main:app --app-dir backend --reload
```

Frontend in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_BASE_URL` to the backend URL if it is not `http://localhost:8000`. On local HTTP, leave `MEDHA_COOKIE_SECURE=false`; use HTTPS and set it to `true` behind a production TLS proxy.

## Test and build

```powershell
$env:PYTHONPATH = "backend"
python -m pytest backend/tests
cd frontend
npm run lint
npm run build
```

Backend integration tests use an isolated SQLite database and a stubbed stream adapter, so they do not pretend to verify real model inference. Run the app with the configured provider to verify actual inference. `docker compose up` applies Alembic migrations before starting the API.

## API and configuration

See [API reference](docs/api.md), [architecture](docs/architecture.md), and [development notes](docs/development.md). Main endpoints include `/api/auth/*`, `/api/conversations`, `/api/projects`, `/api/memories`, `/api/models`, `/health/live`, and `/health/ready`.

Key environment variables are documented in `.env.example`: database URL, auth secret, cookie/CORS settings, provider, model registry, API key, embedding model/dimension, and upload limit.

## Current limitations and next milestones

- This is a functional first release, not a hosted multi-tenant service: add managed secret rotation, distributed rate limiting, backups, observability, and production deployment hardening before exposing it publicly.
- Files are stored as extracted text in PostgreSQL rather than object storage; large knowledge bases need object storage and background indexing jobs.
- Embeddings are generated during upload when a compatible embedding endpoint is configured. If unavailable, retrieval falls back to lexical ranking.
- Research results come from Crossref metadata; Medha does not fetch paywalled papers or claim to verify paper contents.
- Engineering calculations are model-assisted explanations; there is no unrestricted Python execution or simulation runner.
- No mobile/desktop native clients, organization roles, password reset/email verification, or code execution sandbox are included. The responsive web workspace works in mobile browsers; future native targets can share backend APIs.
- Generated code and diagrams live in chat messages for now; there is no editable project file editor, terminal, test runner, Git integration, or saved artifact browser yet.
- A deployment pipeline, external e2e browser suite, and production monitoring remain follow-up work.

Recommended next milestones: production auth/rate limiting, background document indexing and object storage, retrieval evaluations, browser e2e coverage, sandboxed user-approved code execution, and native clients if desired.
