# Dependency decisions

No new Python package is introduced for the launch foundation. The web server uses the repository's existing FastAPI/Uvicorn dependencies. Search networking, model-compatible streaming, password hashing, and SQLite use the Python standard library. CPU model code uses the repository's existing PyTorch dependency.

Existing `requirements.txt` entries predate this launch work and are not newly introduced dependencies. Any future added package must be pinned to an exact version here and in the relevant requirements file, with its purpose and maintenance trade-off documented.
