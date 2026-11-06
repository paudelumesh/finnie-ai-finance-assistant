import io
import sys

import pytest

from src.core import auth
from src.core.auth import (
    AuthConfig, AuthConfigError, LoginThrottle, check_credentials, hash_password, new_session,
    session_valid, verify_password,
)

PW = "correct horse battery"


@pytest.fixture(scope="module")
def cfg():
    return AuthConfig.from_values("Owner@Example.com ", hash_password(PW))


def test_hash_is_salted_self_describing_and_hides_password():
    a, b = hash_password(PW), hash_password(PW)
    assert a != b  # random salt
    assert a.startswith("scrypt$16384$8$1$") and PW not in a
    assert verify_password(PW, a) and verify_password(PW, b)


def test_wrong_password_and_unicode():
    h = hash_password("pässwörd-ünïcode")
    assert verify_password("pässwörd-ünïcode", h)
    assert not verify_password("passwörd-ünïcode", h) and not verify_password("", h)


def test_short_password_rejected():
    with pytest.raises(ValueError, match="at least"):
        hash_password("short")


@pytest.mark.parametrize("bad", ["", "plaintext", "md5$1$2$3$a$b", "scrypt$x$8$1$AAAA$AAAA", "scrypt$1$2$3$!!!$???"])
def test_malformed_hash_is_a_config_error(bad):
    with pytest.raises(AuthConfigError):
        verify_password("x", bad)


@pytest.mark.parametrize("email, h", [(None, "x"), ("", "x"), ("  ", "x"), ("a@b.c", None), ("a@b.c", ""), ("a@b.c", "  ")])
def test_missing_config_fails_closed(email, h):
    with pytest.raises(AuthConfigError, match="not configured"):
        AuthConfig.from_values(email, h)


def test_malformed_hash_rejected_at_config_time():
    with pytest.raises(AuthConfigError, match="malformed"):
        AuthConfig.from_values("a@b.c", "garbage")


def test_check_credentials_requires_both(cfg):
    assert check_credentials("owner@example.com", PW, cfg)
    assert check_credentials("  OWNER@example.COM", PW, cfg)  # email case/whitespace-insensitive
    assert not check_credentials("owner@example.com", PW + "x", cfg)
    assert not check_credentials("other@example.com", PW, cfg)
    assert not check_credentials("", "", cfg) and not check_credentials(None, None, cfg)


def test_wrong_email_still_runs_the_password_hash(cfg, monkeypatch):
    calls = []
    real = auth.verify_password
    monkeypatch.setattr(auth, "verify_password", lambda p, h: calls.append(1) or real(p, h))
    assert not check_credentials("other@example.com", PW, cfg)
    assert calls == [1]  # no early exit that would leak which field was wrong via timing


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_lockout_after_max_failures_then_expires():
    clock = Clock()
    t = LoginThrottle(clock, max_failures=3, window=60, lockout=100)
    assert t.seconds_locked() == 0
    t.record_failure(); t.record_failure()
    assert t.seconds_locked() == 0
    t.record_failure()
    assert t.seconds_locked() == 100
    clock.t = 40
    assert t.seconds_locked() == 60
    clock.t = 101
    assert t.seconds_locked() == 0


def test_failures_outside_window_do_not_accumulate_and_success_resets():
    clock = Clock()
    t = LoginThrottle(clock, max_failures=3, window=60, lockout=100)
    t.record_failure(); t.record_failure()
    clock.t = 120
    t.record_failure()  # old two fell out of the window
    assert t.seconds_locked() == 0
    t.record_failure(); t.record_success(); t.record_failure(); t.record_failure()
    assert t.seconds_locked() == 0


def test_session_validity():
    s = new_session("a@b.c", now=1000)
    assert s == {"email": "a@b.c", "at": 1000}
    assert session_valid(s, now=1000 + 60)
    assert not session_valid(s, now=1000 + auth.SESSION_SECONDS)
    assert not session_valid(s, now=500)  # clock going backwards
    assert not session_valid(None) and not session_valid({}) and not session_valid({"email": "x"})
    assert session_valid(new_session("a@b.c"))


# ---- scripts/set_password ---------------------------------------------------------------
def test_update_env_replaces_and_appends_without_touching_other_lines():
    from scripts.set_password import update_env
    text = "OPENAI_API_KEY=sk-1\nAUTH_EMAIL=old@x.com\nTAVILY_API_KEY=t\n"
    out = update_env(text, {"AUTH_EMAIL": "new@x.com", "AUTH_PASSWORD_HASH": "h"})
    assert out == "OPENAI_API_KEY=sk-1\nAUTH_EMAIL=new@x.com\nTAVILY_API_KEY=t\nAUTH_PASSWORD_HASH=h\n"
    assert update_env("", {"A": "1"}) == "A=1\n"


def test_set_password_writes_env_and_never_the_password(tmp_path, monkeypatch, capsys):
    from scripts import set_password
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-keep\n")
    monkeypatch.setattr(set_password, "ENV_PATH", env)
    monkeypatch.setattr(sys, "argv", ["x", "--email", "Me@Example.com", "--stdin", "--write-env"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("a long enough password\n"))
    set_password.main()
    text = env.read_text()
    assert "OPENAI_API_KEY=sk-keep" in text and "AUTH_EMAIL=me@example.com" in text
    assert "a long enough password" not in text and "a long enough password" not in capsys.readouterr().out
    stored = [l for l in text.splitlines() if l.startswith("AUTH_PASSWORD_HASH=")][0].split("=", 1)[1]
    assert verify_password("a long enough password", stored)


def test_set_password_prints_values_without_write_env_and_rejects_short(tmp_path, monkeypatch, capsys):
    from scripts import set_password
    monkeypatch.setattr(set_password, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(sys, "argv", ["x", "--email", "a@b.c", "--stdin"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("long enough password"))
    set_password.main()
    out = capsys.readouterr().out
    assert 'AUTH_EMAIL = "a@b.c"' in out and "AUTH_PASSWORD_HASH = \"scrypt$" in out
    assert not (tmp_path / ".env").exists()
    monkeypatch.setattr(sys, "stdin", io.StringIO("short"))
    with pytest.raises(SystemExit, match="at least"):
        set_password.main()
