# Capability suite

The bank contains 210 original editable tasks. It is not a copy of a public benchmark. Category counts and checker names are validated by the test suite.

Run it against the configured local brain:

```powershell
python -m evals.capability --brain openai_compat --model <local-model-tag> --out reports/<local-model-tag>/
```

Set `MEDHA_BRAIN_BASE_URL` (for example an Ollama OpenAI-compatible endpoint), and keep the model server running. The model-specific `metrics.json`, `report.md`, 30-row `human_review.csv`, and cross-model `comparison.md` are written under `reports/`. A stub run only validates the harness and is not capability evidence.

The code checker launches short-lived subprocesses for Python and JavaScript, uses an in-memory read-only SQLite fixture for SQL, and parses HTML/CSS with the standard library. Python code imports are allowlisted and network-related imports are denied; POSIX runs also have address-space, CPU, and file-size limits. JavaScript uses a separate Node process and `vm` context. This is a local evaluation harness, not a hardened multi-tenant code execution service; do not run arbitrary submissions from untrusted users through it.

External JSONL tasks may be loaded with `load_public_jsonl(path, licence=...)`. The loader requires a licence note on every import; retain the dataset's source and licence terms with the local copy, and do not commit downloaded datasets unless their terms allow redistribution.
