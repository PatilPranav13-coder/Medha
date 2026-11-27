from __future__ import annotations

import unittest

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from backend.auth_routes import CSRF_COOKIE, SESSION_COOKIE, _check_csrf, _set_session_cookies
from backend.config import Settings
from backend.database import token_digest


def make_request(headers: list[tuple[bytes, bytes]]) -> Request:
    return Request({
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/api/auth/signup",
        "raw_path": b"/api/auth/signup",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 443),
        "root_path": "",
    })


class AuthRouteSecurityTests(unittest.TestCase):
    def test_anonymous_mutation_requires_matching_double_submit_token(self):
        token = "csrf-token-value"
        request = make_request([
            (b"cookie", f"{CSRF_COOKIE}={token}".encode()),
            (b"x-csrf-token", token.encode()),
        ])
        _check_csrf(request, None)
        mismatch = make_request([
            (b"cookie", f"{CSRF_COOKIE}={token}".encode()),
            (b"x-csrf-token", b"different"),
        ])
        with self.assertRaises(HTTPException) as raised:
            _check_csrf(mismatch, None)
        self.assertEqual(raised.exception.status_code, 403)

    def test_signed_in_mutation_checks_hash_from_session_row(self):
        token = "csrf-token-value"
        request = make_request([(b"x-csrf-token", token.encode())])
        _check_csrf(request, {"csrf_hash": token_digest(token)})
        with self.assertRaises(HTTPException):
            _check_csrf(request, {"csrf_hash": token_digest("other")})

    def test_session_cookie_is_secure_httponly_and_same_site_lax(self):
        response = Response()
        _set_session_cookies(response, "bearer-token", "csrf-token", Settings())
        headers = [value.decode("latin-1") for name, value in response.raw_headers if name.lower() == b"set-cookie"]
        session = next(value for value in headers if value.startswith(SESSION_COOKIE + "="))
        csrf = next(value for value in headers if value.startswith(CSRF_COOKIE + "="))
        self.assertIn("HttpOnly", session)
        self.assertIn("Secure", session)
        self.assertIn("SameSite=lax", session)
        self.assertIn("Secure", csrf)
        self.assertIn("SameSite=lax", csrf)
        self.assertNotIn("HttpOnly", csrf)


if __name__ == "__main__":
    unittest.main()
