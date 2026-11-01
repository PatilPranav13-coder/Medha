# API reference

FastAPI serves interactive OpenAPI docs at `/docs`. Authentication is cookie based: send `medha_session` automatically by using same-site requests or browser requests with credentials enabled. The token expires after 12 hours.

## Authentication

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/auth/register` | Create account (`email`, `password`, minimum 10 characters) and set session cookie. |
| POST | `/api/auth/login` | Verify credentials and set session cookie. |
| POST | `/api/auth/logout` | Clear session cookie. |
| GET | `/api/auth/me` | Return current account. |

## Workspace

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/models` | Configured provider and model registry. Requires login. |
| GET | `/api/conversations?q=&project_id=&limit=&offset=` | List the signed-in user's conversations. |
| POST | `/api/conversations` | Create a conversation. Accepts `title`, `project_id`, `mode`, `model_name`. |
| GET/PATCH/DELETE | `/api/conversations/{id}` | Read, rename (`title`), or delete an owned conversation. |
| GET | `/api/conversations/{id}/messages` | Read persisted messages and source metadata. |
| POST | `/api/conversations/{id}/messages` | Start a streamed turn. Accepts `content`, optional `mode`, `model_name`, `web_search`, and `build_target` (`auto`, `apple`, `android`, `windows`, or `cross_platform`). |
| PATCH | `/api/conversations/{id}/messages/{message_id}` | Edit an owned user message. |
| POST | `/api/conversations/{id}/messages/{message_id}/resend` | Remove later turns and regenerate from an edited user message. |
| POST | `/api/conversations/{id}/messages/{message_id}/regenerate` | Regenerate an assistant response to the preceding user message. |
| GET/POST | `/api/projects` | List or create project workspace. |
| PATCH/DELETE | `/api/projects/{id}` | Update or delete an owned project. |
| GET/POST | `/api/projects/{id}/documents` | List or upload a project file as multipart field `file`. |
| DELETE | `/api/projects/{id}/documents/{document_id}` | Delete an owned project file and its chunks. |
| GET/POST | `/api/memories` | List or create user/project memory. |
| DELETE | `/api/memories/{id}` | Delete an owned memory. |
| GET | `/health/live`, `/health/ready` | Liveness and database readiness checks. |

## Streaming

The message endpoint returns `text/event-stream` frames. Each frame is `data: {json}\n\n` and contains one of:

- `{"delta":"..."}` for generated text.
- `{"done":true,"message_id":"...","model":"...","sources":[...]}` after the assistant message is persisted.
- `{"error":"..."}` if the provider fails. The user message remains in history; no fake assistant answer is stored.

Sources include `title`, `url`, `snippet`, `kind`, and optional attribution/academic metadata. Project file citations use `document://` links to the source document/chunk. Public search is opt-in per message and sends that message's prompt to MDN, Microsoft Learn, Stack Exchange, and (in Research mode) Crossref.

## Modes

`code`, `research`, `architecture`, and `engineering` each use different model instructions. Retrieval includes only the active project's files, plus matching user-level and active-project memories. A conversation without an active project does not pull documents from other projects.
