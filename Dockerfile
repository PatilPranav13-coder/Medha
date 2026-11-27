ARG PYTHON_IMAGE=python:3.11.18-slim-bookworm

FROM ${PYTHON_IMAGE} AS dependencies
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY backend/requirements.txt ./requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM ${PYTHON_IMAGE} AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEDHA_DATABASE_PATH=/data/medha.sqlite3
RUN groupadd --system --gid 10001 medha \
    && useradd --system --uid 10001 --gid medha --home-dir /app medha \
    && mkdir -p /app /data \
    && chown medha:medha /data
WORKDIR /app
COPY --from=dependencies /opt/venv /opt/venv
COPY backend ./backend
COPY medha_local_llm ./medha_local_llm
COPY prompts ./prompts
COPY safety ./safety
COPY websearch ./websearch
COPY scripts ./scripts
COPY index.html ./index.html
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
