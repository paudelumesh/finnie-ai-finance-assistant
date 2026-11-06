"""Single-owner authentication: salted scrypt password hash, lockout and session expiry.

There is deliberately no sign-up. The owner's email and password *hash* come from the
environment (.env locally, Streamlit Secrets when deployed); the plaintext password is never
stored. Pure logic only (no Streamlit) so it is easy to test.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable

SCRYPT_N, SCRYPT_R, SCRYPT_P, KEY_LEN = 2**14, 8, 1, 32
SESSION_SECONDS = 8 * 3600
MAX_FAILURES = 5
FAILURE_WINDOW_SECONDS = 15 * 60
LOCKOUT_SECONDS = 15 * 60
MIN_PASSWORD_LENGTH = 8


class AuthConfigError(RuntimeError):
    """Authentication is not (correctly) configured. The app must stay locked."""


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return 'scrypt$N$r$p$salt$hash' (self-describing, so parameters can change later)."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=KEY_LEN)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def _parse(stored: str) -> tuple[int, int, int, bytes, bytes]:
    try:
        scheme, n, r, p, salt, digest = stored.strip().split("$")
        if scheme != "scrypt":
            raise ValueError("unsupported scheme")
        return int(n), int(r), int(p), base64.b64decode(salt, validate=True), base64.b64decode(digest, validate=True)
    except Exception as exc:
        raise AuthConfigError("AUTH_PASSWORD_HASH is malformed; regenerate it with scripts/set_password.py") from exc


def verify_password(password: str, stored_hash: str) -> bool:
    n, r, p, salt, expected = _parse(stored_hash)
    actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=len(expected),
                            maxmem=256 * 1024 * 1024)
    return hmac.compare_digest(actual, expected)


@dataclass(frozen=True)
class AuthConfig:
    email: str
    password_hash: str

    @classmethod
    def from_values(cls, email: str | None, password_hash: str | None) -> "AuthConfig":
        """Fail closed: missing or malformed configuration raises, it never means 'no login'."""
        missing = [name for name, value in (("AUTH_EMAIL", email), ("AUTH_PASSWORD_HASH", password_hash))
                   if not value or not value.strip()]
        if missing:  # names only, never values or other key names (this text is shown to anonymous visitors)
            raise AuthConfigError(
                f"Login is not configured: {' and '.join(missing)} not found in the app's secrets. "
                "Set AUTH_EMAIL and AUTH_PASSWORD_HASH (create the hash with `python -m scripts.set_password`)."
            )
        _parse(password_hash)  # validates the format now rather than at first login
        return cls(email.strip().lower(), password_hash.strip())


def check_credentials(email: str, password: str, cfg: AuthConfig) -> bool:
    """Always runs the password hash, even for a wrong email, so timing doesn't reveal which was wrong."""
    email_ok = hmac.compare_digest((email or "").strip().lower().encode(), cfg.email.encode())
    password_ok = verify_password(password or "", cfg.password_hash)
    return email_ok and password_ok


class LoginThrottle:
    """Process-wide lockout: MAX_FAILURES failures inside the window locks logins for LOCKOUT_SECONDS.

    Global (not per session) on purpose: an attacker can always open a fresh browser session.
    State is in memory, so it resets if the app restarts.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic, max_failures: int = MAX_FAILURES,
                 window: float = FAILURE_WINDOW_SECONDS, lockout: float = LOCKOUT_SECONDS) -> None:
        self._clock, self.max_failures, self.window, self.lockout = clock, max_failures, window, lockout
        self._failures: list[float] = []
        self._locked_until = 0.0
        self._lock = threading.Lock()

    def seconds_locked(self) -> int:
        with self._lock:
            return max(0, int(self._locked_until - self._clock() + 0.999))

    def record_failure(self) -> None:
        with self._lock:
            now = self._clock()
            self._failures = [t for t in self._failures if now - t < self.window] + [now]
            if len(self._failures) >= self.max_failures:
                self._locked_until = now + self.lockout
                self._failures.clear()

    def record_success(self) -> None:
        with self._lock:
            self._failures.clear()


def new_session(email: str, now: float | None = None) -> dict:
    return {"email": email, "at": time.time() if now is None else now}


def session_valid(auth: dict | None, now: float | None = None, max_age: float = SESSION_SECONDS) -> bool:
    if not auth or "at" not in auth:
        return False
    return 0 <= (time.time() if now is None else now) - auth["at"] < max_age
