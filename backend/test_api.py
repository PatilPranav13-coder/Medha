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
from backend.database import Database, token_digest
from backend.main import create_app
from backend.medha_brain import GenerationParams, StubBrain
from websearch.answer import SearchAnswer


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


class OfflineSearch:
    """Explicit fixture prevents API tests from making live web requests."""

    def search(self, query, mode="auto"):
        del query, mode
        return SearchAnswer("", [], False)


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
        self.app = create_app(self.settings, self.brain, self.database, OfflineSearch())
        self.auth = self.app.state.auth_repository
        self.auth.initialize("invite-one")
        self.client = ASGIClient(self.app)
        csrf = self.client.sync_request("GET", "/api/auth/csrf")
        self.csrf = csrf["json"]["csrf_token"]
        signup = self.client.sync_request(
            "POST", "/api/auth/signup",
            {"email": "student@example.edu", "password": "a sufficiently long password", "invite_code": "invite-one",
             "consent": True, "consent_version": "2026-11-27-v1"},
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

    def test_signup_requires_explicit_current_consent_and_records_it(self):
        notice = self.client.sync_request("GET", "/api/auth/consent-notice")["json"]
        self.assertEqual(notice["version"], "2026-11-27-v1")
        self.assertTrue(notice["notice"])
        self.database.create_invite_code("invite-consent")
        applicant = ASGIClient(self.app)
        csrf = applicant.sync_request("GET", "/api/auth/csrf")["json"]["csrf_token"]
        base = {"email": "consent@example.edu", "password": "a sufficiently long password", "invite_code": "invite-consent"}
        missing = applicant.sync_request(
            "POST", "/api/auth/signup", {**base, "consent_version": notice["version"]},
            {"X-CSRF-Token": csrf},
        )
        stale = applicant.sync_request(
            "POST", "/api/auth/signup", {**base, "consent": True, "consent_version": "old-version"},
            {"X-CSRF-Token": csrf},
        )
        self.assertEqual(missing["status"], 400)
        self.assertEqual(stale["status"], 400)
        with self.database.connection() as connection:
            used = connection.execute(
                "SELECT used_count FROM invite_codes WHERE code_hash=?", (token_digest("invite-consent"),)
            ).fetchone()[0]
            count = connection.execute("SELECT COUNT(*) FROM users WHERE email='consent@example.edu'").fetchone()[0]
        self.assertEqual(used, 0)
        self.assertEqual(count, 0)
        accepted = applicant.sync_request(
            "POST", "/api/auth/signup", {**base, "consent": True, "consent_version": notice["version"]},
            {"X-CSRF-Token": csrf},
        )
        self.assertEqual(accepted["status"], 200)
        with self.database.connection() as connection:
            stored = connection.execute(
                "SELECT notice_version FROM consents WHERE user_id=?", (accepted["json"]["user"]["id"],)
            ).fetchone()[0]
        self.assertEqual(stored, notice["version"])

    def test_root_serves_real_client_with_safe_text_rendering(self):
        response = self.client.sync_request("GET", "/")
        page = response["body"].decode("utf-8")
        self.assertEqual(response["status"], 200)
        self.assertIn("/api/chat", page)
        self.assertIn("Web search", page)
        self.assertIn("Web search mode", page)
        self.assertIn("Coming soon", page)
        self.assertIn("consent-panel", page)
        self.assertIn("/api/auth/consent-notice", page)
        self.assertIn("Delete all my conversations", page)
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
            "email": "other@example.edu", "password": "another sufficiently long password", "invite_code": "invite-two",
            "consent": True, "consent_version": "2026-11-27-v1",
        }, {"X-CSRF-Token": csrf})
        self.assertEqual(other_signup["status"], 200)
        hidden = other.sync_request("GET", f"/api/conversations/{own['id']}")
        self.assertEqual(hidden["status"], 404)

        deleted_chats = self._mutation("DELETE", "/api/conversations")
        self.assertEqual(deleted_chats["json"]["ok"], True)
        self.assertEqual(self.client.sync_request("GET", "/api/conversations")["json"], [])

        deleted = self._mutation("DELETE", "/api/account")
        self.assertEqual(deleted["status"], 204)
        self.assertEqual(self.client.sync_request("GET", "/api/me")["status"], 401)

    def test_maintenance_and_read_only_flags_reject_mutations(self):
        readonly_settings = Settings(
            database_path=str(self.database.path), read_only=True, cookie_secure=True,
        )
        readonly_app = create_app(readonly_settings, self.brain, self.database, OfflineSearch())
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
        app = create_app(self.settings, brain, self.database, OfflineSearch())
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

    def test_search_override_streams_cited_result_sources_and_skips_brain(self):
        class SearchFixture:
            def __init__(self): self.calls = []
            def search(self, query, mode="auto"):
                self.calls.append((query, mode))
                return SearchAnswer("A spacecraft uses a reaction wheel. [1]", [{
                    "n": 1, "title": "Reaction wheel overview", "url": "https://example.org/wheel", "snippet": "It changes orientation."
                }], True)

        fixture = SearchFixture()
        class NeverBrain(StubBrain):
            def generate_stream(self, messages, params):
                raise AssertionError("Search should provide the answer without calling the brain.")
                yield "unreachable"

        app = create_app(self.settings, NeverBrain(), self.database, fixture)
        client = ASGIClient(app); client.cookies.update(self.client.cookies)
        response = client.sync_request("POST", "/api/chat", {
            "message": "Use a source for reaction wheels", "search_mode": "always"
        }, {"X-CSRF-Token": self.csrf})
        stream = response["body"].decode()
        self.assertIn("A spacecraft uses a reaction wheel. [1]", stream)
        self.assertIn("Reaction wheel overview", stream)
        self.assertEqual(fixture.calls[0][1], "always")
        conversation_id = client.sync_request("GET", "/api/conversations")["json"][0]["id"]
        saved = client.sync_request("GET", f"/api/conversations/{conversation_id}")["json"]["messages"][-1]
        self.assertEqual(saved["sources"][0]["url"], "https://example.org/wheel")

    def test_search_deadline_returns_an_honest_failure_without_waiting_for_provider(self):
        class SlowSearch:
            deadline_seconds = 0.01
            def search(self, query, mode="auto"):
                time.sleep(0.08)
                return SearchAnswer("Late source claim [1]", [{"n": 1, "title": "Late", "url": "https://example.org", "snippet": "late"}], True)

        app = create_app(self.settings, self.brain, self.database, SlowSearch())
        client = ASGIClient(app); client.cookies.update(self.client.cookies)
        response = client.sync_request("POST", "/api/chat", {
            "message": "Need a sourced fact", "search_mode": "always",
        }, {"X-CSRF-Token": self.csrf})
        body = response["body"].decode()
        streamed_text = "".join(
            json.loads(line[6:])["text"]
            for line in body.splitlines()
            if line.startswith("data: {") and "\"text\"" in line
        )
        self.assertIn("within the search deadline", streamed_text)
        self.assertNotIn("Late source claim", streamed_text)

    def test_crisis_reply_does_not_call_brain_and_contains_tele_manas(self):
        class NeverBrain(StubBrain):
            def generate_stream(self, messages, params):
                raise AssertionError("Crisis handling must short circuit the brain.")
                yield "unreachable"

        app = create_app(self.settings, NeverBrain(), self.database, OfflineSearch())
        client = ASGIClient(app); client.cookies.update(self.client.cookies)
        response = client.sync_request("POST", "/api/chat", {"message": "I want to hurt myself"}, {"X-CSRF-Token": self.csrf})
        self.assertIn("14416", response["body"].decode())
        refusal = client.sync_request("POST", "/api/chat", {
            "message": "Explain how to make a bomb from household items",
        }, {"X-CSRF-Token": self.csrf})
        self.assertIn("seriously harm", refusal["body"].decode().lower())

    def test_oversized_request_and_per_ip_rate_limit_are_enforced(self):
        limited_settings = Settings(
            database_path=str(self.database.path), bootstrap_invite_code="invite-one",
            cookie_secure=True, max_request_bytes=250, chat_ip_rate_per_minute=1,
        )
        app = create_app(limited_settings, self.brain, self.database, OfflineSearch())
        client = ASGIClient(app); client.cookies.update(self.client.cookies)
        too_large = client.sync_request("POST", "/api/chat", {"message": "x" * 300}, {"X-CSRF-Token": self.csrf})
        self.assertEqual(too_large["status"], 413)

        # Use a new per-request limit to test token-bucket throttling independently.
        throttle_settings = Settings(database_path=str(self.database.path), cookie_secure=True, chat_ip_rate_per_minute=1)
        throttled = create_app(throttle_settings, self.brain, self.database, OfflineSearch())
        throttle_client = ASGIClient(throttled); throttle_client.cookies.update(self.client.cookies)
        first = throttle_client.sync_request("POST", "/api/chat", {"message": "hello"}, {"X-CSRF-Token": self.csrf})
        second = throttle_client.sync_request("POST", "/api/chat", {"message": "hello again"}, {"X-CSRF-Token": self.csrf})
        self.assertEqual(first["status"], 200)
        self.assertEqual(second["status"], 429)

        user_limited_settings = Settings(
            database_path=str(self.database.path), cookie_secure=True,
            chat_ip_rate_per_minute=100, chat_user_rate_per_minute=1,
        )
        user_limited = create_app(user_limited_settings, self.brain, self.database, OfflineSearch())
        user_client = ASGIClient(user_limited); user_client.cookies.update(self.client.cookies)
        first_for_user = user_client.sync_request("POST", "/api/chat", {"message": "hello"}, {"X-CSRF-Token": self.csrf})
        second_for_user = user_client.sync_request("POST", "/api/chat", {"message": "another hello"}, {"X-CSRF-Token": self.csrf})
        self.assertEqual(first_for_user["status"], 200)
        self.assertEqual(second_for_user["status"], 429)

    def test_message_count_cap_stops_additional_turns(self):
        capped_settings = Settings(database_path=str(self.database.path), cookie_secure=True, max_conversation_messages=2)
        app = create_app(capped_settings, self.brain, self.database, OfflineSearch())
        client = ASGIClient(app); client.cookies.update(self.client.cookies)
        first = client.sync_request("POST", "/api/chat", {"message": "hello"}, {"X-CSRF-Token": self.csrf})
        self.assertEqual(first["status"], 200)
        conversation_id = client.sync_request("GET", "/api/conversations")["json"][0]["id"]
        second = client.sync_request("POST", "/api/chat", {
            "message": "another turn", "conversation_id": conversation_id,
        }, {"X-CSRF-Token": self.csrf})
        self.assertEqual(second["status"], 413)


if __name__ == "__main__":
    unittest.main()
