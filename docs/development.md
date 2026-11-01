# Development notes

## Local setup

1. Copy `.env.example` to `.env`, replace the auth and database secrets, and use host service URLs if running the API outside Compose.
2. Start PostgreSQL with pgvector or leave `MEDHA_DATABASE_URL` unset for SQLite. Start Ollama or configure an OpenAI-compatible endpoint.
3. Install `backend/requirements-dev.txt`, set `PYTHONPATH=backend`, apply migrations with `python -m alembic -c backend/alembic.ini upgrade head`, then run `uvicorn app.main:app --app-dir backend --reload`.
4. In `frontend/`, run `npm install` and `npm run dev`.

For Compose, `docker compose up --build -d` runs the database, Ollama, migration, API, and web application. Pull the configured chat and embedding models into the Ollama volume explicitly.

## Quality checks

Run `python -m pytest backend/tests`, `npm run lint`, and `npm run build`. Backend integration tests use temporary SQLite and replace the model stream for persistence paths; they do not certify a real provider connection. A live model smoke test requires Ollama or a valid configured endpoint. Compose should also be exercised in an environment with Docker installed.

## Database migrations

The versioned schema lives in `backend/alembic/versions`. Use `python -m alembic -c backend/alembic.ini upgrade head` to migrate. PostgreSQL creates the `vector` extension before the first migration. FastAPI startup also calls `create_all` for quick SQLite development; production upgrades should use Alembic.

## Adding a mode or provider

Add a mode to `Mode` in `backend/app/schemas.py`, add mode-specific instructions in `backend/app/api/conversations.py`, wire it into the frontend mode picker, and extend integration coverage. Keep provider requests behind `backend/app/services/ai.py`; model and search failures should be surfaced rather than replaced by canned responses. Do not execute generated source code inside the API process.

## Security notes

- Keep `.env` out of version control. Configure HTTPS, secure cookies, and explicit CORS origins at deployment.
- Use a URL-safe database password with the Compose connection string, or percent-encode reserved URL characters.
- The current session token is signed and expires, but there is no refresh-token/session-revocation table yet.
- The API applies per-process IP limits to login/registration and chat generation, and checks browser request origins. These limits reset on restart and are not shared across workers; replace/augment them with distributed limits at the trusted reverse proxy before public production deployment.
- Add centralized logging/metrics, backups, and secret rotation before public production deployment.
- Uploaded text and search excerpts are untrusted prompt context. The API scopes retrieval and CRUD operations to the authenticated account and never executes uploads.
