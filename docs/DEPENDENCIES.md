# Dependency decisions

The web server uses the repository's existing FastAPI/Uvicorn dependencies. Search networking, model-compatible streaming, password hashing, and SQLite use the Python standard library. CPU model code uses the repository's existing PyTorch dependency.

| Package | Pin | Purpose | Trade-off |
|---|---|---|---|
| `sentry-sdk` | `2.77.0` | Optional exception reporting when `SENTRY_DSN` is set. PII capture, request bodies and local variables are disabled, and a scrubber removes credentials and query strings. | The owner must operate a Sentry project and review its retention/access settings. It is absent from the data path when the DSN is unset. The owner authorized this addition. |
| Ruff | `0.17.6` | CI static lint check for Python code. | Adds a pinned CI-only binary tool. |
| pip-audit | `2.10.1` | CI check for known vulnerabilities in the resolved Python requirements. | Requires a vulnerability database lookup during CI. |

Container image tags are pinned in the build files: Python `3.11.18-slim-bookworm`, Caddy `2.11.9`, and optional Ollama `0.40.18`. Review and update these tags deliberately when applying security updates.

Existing `requirements.txt` entries predate this launch work and are not newly introduced dependencies. CI tooling is pinned in the workflow: Ruff `0.17.6` and pip-audit `2.10.1`. Any future added package must be pinned to an exact version here and in the relevant requirements file, with its purpose and maintenance trade-off documented.
