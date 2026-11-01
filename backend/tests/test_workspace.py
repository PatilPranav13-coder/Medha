from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient


def test_browser_mutation_rejects_untrusted_origin(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={"email": "origin@example.com", "password": "strong-test-password"},
        headers={"Origin": "https://untrusted.example"},
    )
    assert response.status_code == 403


def test_authentication_and_private_project_boundary(client: TestClient) -> None:
    first = client.post("/api/auth/register", json={"email": "one@example.com", "password": "first-secure-pass"})
    assert first.status_code == 201
    assert first.cookies.get("medha_session")
    assert first.headers.get("set-cookie", "").lower().find("httponly") >= 0

    project = client.post("/api/projects", json={"name": "Private work", "description": "Only me"})
    assert project.status_code == 201
    project_id = project.json()["id"]
    client.post("/api/auth/logout")

    second = TestClient(client.app)
    signup = second.post("/api/auth/register", json={"email": "two@example.com", "password": "second-secure-pass"})
    assert signup.status_code == 201
    assert second.get(f"/api/projects/{project_id}/documents").status_code == 404


def test_project_upload_memory_and_message_history(signed_in: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api import conversations
    observed_system_prompts: list[str] = []

    async def fake_stream(_messages: list[dict[str, str]], _model: str, on_usage=None) -> AsyncIterator[str]:
        observed_system_prompts.append(_messages[0]["content"])
        if on_usage:
            on_usage(18, 7)
        yield "A clear explanation"
        yield " with a code example."

    monkeypatch.setattr(conversations, "stream_chat", fake_stream)
    project = signed_in.post("/api/projects", json={"name": "Medha Core", "instructions": "Prefer Python."}).json()
    uploaded = signed_in.post(
        f"/api/projects/{project['id']}/documents",
        files={"file": ("notes.md", b"The API uses FastAPI and SQLAlchemy.", "text/markdown")},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["chunk_count"] == 1

    memory = signed_in.post("/api/memories", json={"label": "Preferred language", "content": "Use Python for backend examples.", "project_id": project["id"]})
    assert memory.status_code == 201

    conversation = signed_in.post("/api/conversations", json={"title": "New conversation", "project_id": project["id"], "mode": "code"}).json()
    result = signed_in.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={"content": "Explain this project", "mode": "code", "web_search": False, "build_target": "apple"},
    )
    assert result.status_code == 200
    assert '"delta": "A clear explanation"' in result.text
    assert '"done": true' in result.text

    messages = signed_in.get(f"/api/conversations/{conversation['id']}/messages").json()
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[-1]["content"] == "A clear explanation with a code example."
    assert (messages[-1]["input_tokens"], messages[-1]["output_tokens"]) == (18, 7)
    assert "iPhone, iPad, and Mac" in observed_system_prompts[0]
    assert "FastAPI and SQLAlchemy" in observed_system_prompts[0]
    assert "Preferred language" in observed_system_prompts[0]
    assert signed_in.get("/api/projects").json()[0]["name"] == "Medha Core"
    assert signed_in.get("/api/memories").json()[0]["label"] == "Preferred language"


def test_edit_and_regenerate_message(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api import conversations

    async def fake_stream(messages: list[dict[str, str]], _model: str, on_usage=None) -> AsyncIterator[str]:
        yield "Regenerated: " + messages[-1]["content"]

    monkeypatch.setattr(conversations, "stream_chat", fake_stream)
    client.post("/api/auth/register", json={"email": "edit@example.com", "password": "edit-test-password"})
    conversation = client.post("/api/conversations", json={"title": "Test"}).json()
    client.post(f"/api/conversations/{conversation['id']}/messages", json={"content": "old question", "web_search": False})
    original = client.get(f"/api/conversations/{conversation['id']}/messages").json()
    user_message, assistant_message = original
    edited = client.patch(
        f"/api/conversations/{conversation['id']}/messages/{user_message['id']}",
        json={"content": "new question"},
    )
    assert edited.status_code == 200
    regenerated = client.post(f"/api/conversations/{conversation['id']}/messages/{assistant_message['id']}/regenerate")
    assert regenerated.status_code == 200
    assert "Regenerated: new question" in regenerated.text


def test_public_lookup_context_is_cited_and_persisted(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api import conversations

    async def fake_search(_query: str) -> list[dict[str, str]]:
        return [{"title": "Python tutorial", "url": "https://docs.python.org/3/tutorial/", "snippet": "Learn Python step by step.", "kind": "documentation"}]

    async def fake_stream(messages: list[dict[str, str]], _model: str, on_usage=None) -> AsyncIterator[str]:
        assert "[S1] Python tutorial" in messages[0]["content"]
        yield "Use the official tutorial. [S1]"

    monkeypatch.setattr(conversations, "search_public_docs", fake_search)
    monkeypatch.setattr(conversations, "stream_chat", fake_stream)
    client.post("/api/auth/register", json={"email": "sources@example.com", "password": "source-test-password"})
    conversation = client.post("/api/conversations", json={"title": "New conversation"}).json()
    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={"content": "Teach me Python", "mode": "code", "web_search": True},
    )
    assert response.status_code == 200
    assert "https://docs.python.org/3/tutorial/" in response.text
    saved = client.get(f"/api/conversations/{conversation['id']}/messages").json()
    assert saved[-1]["sources"][0]["title"] == "Python tutorial"
