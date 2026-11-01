from __future__ import annotations

import os

os.environ["MEDHA_AUTH_SECRET"] = "test-only-secret-for-medha-tests"
os.environ["MEDHA_DATABASE_URL"] = "sqlite:///./test_medha.db"
os.environ["MEDHA_AI_PROVIDER"] = "ollama"
os.environ["MEDHA_AI_BASE_URL"] = "http://ollama.invalid:11434"
os.environ["MEDHA_MODEL_NAME"] = "test-model"
os.environ["MEDHA_EMBEDDING_MODEL"] = ""

import pytest
from fastapi.testclient import TestClient

from app.db import Base, engine
from app.main import app


@pytest.fixture
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def signed_in(client: TestClient) -> TestClient:
    response = client.post("/api/auth/register", json={"email": "dev@example.com", "password": "local-test-password"})
    assert response.status_code == 201
    return client
