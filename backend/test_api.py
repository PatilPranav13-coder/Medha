from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
import unittest
from http.cookies import SimpleCookie
from pathlib import Path

from backend.auth_routes import CSRF_COOKIE, SESSION_COOKIE
from backend.config import Settings
from backend.database import Database
from backend.main import create_app
from backend.medha_brain import GenerationParams, StubBrain


class ASGIClient:
    """Small in-process ASGI driver; it uses no HTTP socket or third-party client."""

    def __init__(self, app):
        self.app = app
        self.cookies: dict[str, str] = {}
        self.disconnect_on_token = False

    async def request(self, method: str, path: str, body: object | None = None, headers: dict[str, str] | None = None):
        request_headers = {key.lower(): value for key, value in (headers or {}).items()}
        if body is not None:
            request_headers.setdefault("content-type", "application/json")
        if self.cookies:
            request_headers["cookie"] = "; ".join(f"{key}={value}" for key, value in self.cookies.items())
        encoded_headers = [(key.encode("latin-1"), value.encode("latin-1")) for key, value in request_headers.items()]
        raw_body = b"" if body is None else json.dumps(body).encode("utf-8")
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": method.upper(), "scheme": "https", "path": path,
            "raw_path": path.encode("utf-8"), "query_string": b"", "headers": encoded_headers,
            "client": ("127.0.0.1", 1000), "server": ("testserver", 443), "root_path": "",
        }
        sent = []
        first = True
        disconnect = asyncio.Event()

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": raw_body, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)
            if self.disconnect_on_token and message.get("type") == "http.response.body" and b"event: token" in message.get("body", b""):
                disconnect.set()

        await self.app(scope, receive, send)
        start = next(message for message in sent if message["type"] == "http.response.start")
        response_headers = start.get("headers", [])
        for name, value in response_headers:
            if name.lower() == b"set-cookie":
                parsed = SimpleCookie()
                parsed.load(value.decode("latin-1"))
                for key, morsel in parsed.items():
                    if morsel["max-age"] == "0":
                        self.cookies.pop(key, None)
                    else:
                        self.cookies[key] = morsel.value
        response_body = b"".join(
            message.get("body", b"") for message in sent if message["type"] == "http.response.body"
        )
        return {
            "status": start["status"],
            "headers": response_headers,
            "body": response_body,
            "json": json.loads(response_body) if response_body and b"application/json" in dict(response_headers).get(b"content-type", b"") else None,
        }

    def sync_request(self, method: str, path: str, body: object | None = None, headers: dict[str, str] | None = None):
        return asyncio.run(self.request(method, path, body, headers))


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temp.name) / "test.sqlite3")
        self.settings = Settings(
            database_path=str(self.database.path),
            bootstrap_invite_code="invite-one",
            cookie_secure=True,
            max_concurrent_generations=1,
            max_queued_generations=1,
        )
        self.brain = StubBrain("A fixed answer with **formatting**.")
        self.app = create_app(self.settings, self.brain, self.database)
        self.auth = self.app.state.auth_repository
        self.auth.initialize("invite-one")
        self.client = ASGIClient(self.app)
        csrf = self.client.sync_request("GET", "/api/auth/csrf")
        self.csrf = csrf["json"]["csrf_token"]
        signup = self.client.sync_request(
            "POST", "/api/auth/signup",
            {"email": "student@example.edu", "password": "a sufficiently long password", "invite_code": "invite-one"},
            {"X-CSRF-Token": self.csrf},
        )
        self.assertEqual(signup["status"], 200)
        self.csrf = signup["json"]["csrf_token"]

    def tearDown(self):
        self.temp.cleanup()

    def _mutation(self, method: str, path: str, body: object | None = None):
        return self.client.sync_request(method, path, body, {"X-CSRF-Token": self.csrf})

    def test_invite_signup_me_and_secure_session_cookie(self):
        self.assertIn(SESSION_COOKIE, self.client.cookies)
        self.assertIn(CSRF_COOKIE, self.client.cookies)
        me = self.client.sync_request("GET", "/api/me")
        self.assertEqual(me["status"], 200)
        self.assertEqual(me["json"]["user"]["email"], "student@example.edu")

    def test_root_serves_real_client_with_safe_text_rendering(self):
        response = self.client.sync_request("GET", "/")
        page = response["body"].decode("utf-8")
        self.assertEqual(response["status"], 200)
        self.assertIn("/api/chat", page)
        self.assertIn("Web search", page)
        self.assertIn("Coming soon", page)
        self.assertNotIn("innerHTML", page)
        self.assertIn("noopener noreferrer", page)

    def test_chat_stream_persists_messages_and_supports_regenerate_and_edit(self):
        response = self._mutation("POST", "/api/chat", {"message": "Explain this idea"})
        self.assertEqual(response["status"], 200)
        text = response["body"].decode("utf-8")
        self.assertIn("event: token", text)
        self.assertIn("event: sources", text)
        self.assertIn("event: done", text)
        listing = self.client.sync_request("GET", "/api/conversations")
        self.assertEqual(len(listing["json"]), 1)
        conversation_id = listing["json"][0]["id"]
        detail = self.client.sync_request("GET", f"/api/conversations/{conversation_id}")
        self.assertEqual([message["role"] for message in detail["json"]["messages"]], ["user", "assistant"])
        self.assertEqual(detail["json"]["messages"][1]["content"], "A fixed answer with **formatting**.")

        regenerated = self._mutation("POST", "/api/chat", {"conversation_id": conversation_id, "action": "regenerate"})
        self.assertIn("event: done", regenerated["body"].decode("utf-8"))
        edited = self._mutation("POST", "/api/chat", {
            "conversation_id": conversation_id, "action": "edit", "message": "A corrected question"
        })
        self.assertIn("event: done", edited["body"].decode("utf-8"))
        final = self.client.sync_request("GET", f"/api/conversations/{conversation_id}")["json"]["messages"]
        self.assertEqual(final[-2]["role"], "user")
        self.assertEqual(final[-2]["content"], "A corrected question")
        self.assertEqual(final[-1]["role"], "assistant")

    def test_state_changing_conversation_request_requires_csrf(self):
        result = self.client.sync_request("POST", "/api/conversations", {"title": "No token"})
        self.assertEqual(result["status"], 403)
        created = self._mutation("POST", "/api/conversations", {"title": "Research"})
        self.assertEqual(created["status"], 200)
        renamed = self._mutation("PATCH", f"/api/conversations/{created['json']['id']}", {"title": "ISRO notes"})
        self.assertEqual(renamed["status"], 200)

    def test_conversation_ownership_feedback_export_and_account_deletion(self):
        stream = self._mutation("POST", "/api/chat", {"message": "Keep this"})
        del stream
        own = self.client.sync_request("GET", "/api/conversations")["json"][0]
        detail = self.client.sync_request("GET", f"/api/conversations/{own['id']}")["json"]
        message_id = detail["messages"][-1]["id"]
        feedback = self._mutation("POST", "/api/feedback", {"message_id": message_id, "rating": 1})
        self.assertEqual(feedback["status"], 200)
        exported = self.client.sync_request("GET", "/api/export")
        self.assertEqual(exported["json"]["email"], "student@example.edu")
        self.assertNotIn("password_hash", json.dumps(exported["json"]))

        self.database.create_invite_code("invite-two")
        other = ASGIClient(self.app)
        csrf = other.sync_request("GET", "/api/auth/csrf")["json"]["csrf_token"]
        other_signup = other.sync_request("POST", "/api/auth/signup", {
            "email": "other@example.edu", "password": "another sufficiently long password", "invite_code": "invite-two"
        }, {"X-CSRF-Token": csrf})
        self.assertEqual(other_signup["status"], 200)
        hidden = other.sync_request("GET", f"/api/conversations/{own['id']}")
        self.assertEqual(hidden["status"], 404)

        deleted = self._mutation("DELETE", "/api/account")
        self.assertEqual(deleted["status"], 204)
        self.assertEqual(self.client.sync_request("GET", "/api/me")["status"], 401)

    def test_maintenance_and_read_only_flags_reject_mutations(self):
        readonly_settings = Settings(
            database_path=str(self.database.path), read_only=True, cookie_secure=True,
        )
        readonly_app = create_app(readonly_settings, self.brain, self.database)
        readonly_client = ASGIClient(readonly_app)
        readonly_client.cookies.update(self.client.cookies)
        response = readonly_client.sync_request(
            "POST", "/api/conversations", {"title": "Blocked"}, {"X-CSRF-Token": self.csrf}
        )
        self.assertEqual(response["status"], 503)

    def test_client_disconnect_closes_brain_stream_and_releases_generation_slot(self):
        class TrackingBrain(StubBrain):
            def __init__(self):
                super().__init__("unused")
                self.closed = threading.Event()

            def generate_stream(self, messages, params):
                del messages, params
                try:
                    yield "first chunk"
                    yield "second chunk"
                finally:
                    self.closed.set()

        brain = TrackingBrain()
        app = create_app(self.settings, brain, self.database)
        client = ASGIClient(app)
        client.cookies.update(self.client.cookies)
        client.disconnect_on_token = True
        result = client.sync_request(
            "POST", "/api/chat", {"message": "stop after first chunk"}, {"X-CSRF-Token": self.csrf}
        )
        self.assertEqual(result["status"], 200)
        self.assertTrue(brain.closed.wait(timeout=1))
        deadline = time.monotonic() + 1
        while app.state.generation_gate.active and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(app.state.generation_gate.active, 0)
        conversation = client.sync_request("GET", "/api/conversations")["json"][0]
        messages = client.sync_request("GET", f"/api/conversations/{conversation['id']}")["json"]["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])


if __name__ == "__main__":
    unittest.main()
