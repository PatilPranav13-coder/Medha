FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend
COPY medha_local_llm /app/medha_local_llm

ENV PORT=8000
ENV MEDHA_CHECKPOINT=/app/medha_local_llm/checkpoints/medha.pt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "8000"]
