"""The login gate, tested through the real app script (AppTest)."""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.core.auth import AuthConfig, LoginThrottle, hash_password, new_session
from src.core.settings import Secrets, Settings, get_settings
from src.data.market_data import MarketDataService, MockProvider
from src.data.news import NewsService, SampleProvider
from src.rag.chunker import chunk_articles
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.web_app import login, runtime
from src.web_app.login import attempt_login
from src.workflow.factory import Services
from tests.test_rag import ARTICLES
from tests.test_web_app import FakeWorkflow

APP = str(Path(__file__).resolve().parents[1] / "src" / "web_app" / "app.py")
EMAIL, PASSWORD = "owner@example.com", "a long test passphrase"
HASH = hash_password(PASSWORD)


def settings_with(email=EMAIL, password_hash=HASH):
    return Settings(secrets=Secrets(auth_email=email, auth_password_hash=password_hash))


@pytest.fixture
def make_app(monkeypatch):
    def build(settings=None):
        st.cache_resource.clear()
        store = VectorStore.build(chunk_articles(ARTICLES, 400, 50), HashEmbedder())
        services = Services(FakeWorkflow(), MarketDataService(MockProvider()), NewsService(SampleProvider()),
                            Retriever(store, 3, 0.1), get_settings())
        monkeypatch.setattr(runtime, "create_runtime", lambda s=None: runtime.Runtime(services, runtime.AsyncRunner()))
        chosen = settings or settings_with()
        monkeypatch.setattr(login, "load_auth_credentials",
                            lambda: (chosen.secrets.auth_email, chosen.secrets.auth_password_hash))
        monkeypatch.setattr(login, "FAILURE_DELAY_SECONDS", 0)
        return AppTest.from_file(APP, default_timeout=30)
    return build


def sign_in(at, email=EMAIL, password=PASSWORD):
    at.text_input[0].set_value(email)
    at.text_input[1].set_value(password)
    at.button[0].click()
    return at.run()


def test_login_form_shown_first_and_app_not_started(make_app):
    at = make_app().run()
    assert not at.exception
    assert [t.label for t in at.text_input] == ["Email", "Password"]
    assert at.text_input[1].proto.type == 1  # PASSWORD: input is masked
    assert len(at.tabs) == 0 and not at.chat_input  # nothing of the app is rendered


def test_no_signup_or_register_path(make_app):
    at = make_app().run()
    labels = [b.label.lower() for b in at.button] + [t.label.lower() for t in at.text_input]
    assert labels == ["sign in", "email", "password"]
    assert not any(w in " ".join(labels) for w in ("sign up", "register", "create account", "forgot"))


def test_correct_login_opens_the_app_then_signout_locks_it(make_app):
    at = sign_in(make_app().run())
    assert not at.exception and "auth" in at.session_state
    assert [t.label for t in at.tabs][:2] == ["Chat", "Portfolio"]
    assert any(EMAIL in c.value for c in at.sidebar.caption)
    next(b for b in at.sidebar.button if b.label == "Sign out").click().run()
    assert "auth" not in at.session_state and [t.label for t in at.text_input] == ["Email", "Password"]


@pytest.mark.parametrize("email, password", [(EMAIL, "wrong password!!"), ("other@example.com", PASSWORD), ("", "")])
def test_bad_credentials_rejected_with_one_generic_message(make_app, email, password):
    at = sign_in(make_app().run(), email, password)
    assert [e.value for e in at.error] == ["Invalid email or password."]
    assert "auth" not in at.session_state and len(at.tabs) == 0


def test_lockout_after_repeated_failures_disables_the_form(make_app):
    at = make_app().run()
    for _ in range(5):
        at = sign_in(at, EMAIL, "wrong password!!")
    assert any("Too many failed attempts" in e.value for e in at.error)  # shown immediately
    at = at.run()  # next interaction: the button now renders disabled
    assert at.button[0].disabled and "auth" not in at.session_state and len(at.tabs) == 0
    assert any("Too many failed attempts" in e.value for e in at.error)


def test_attempt_login_enforces_the_lock_even_for_the_right_password():
    cfg = AuthConfig.from_values(EMAIL, HASH)
    throttle = LoginThrottle(max_failures=2)
    assert attempt_login(EMAIL, "nope nope nope", cfg, throttle) == "invalid"
    assert attempt_login(EMAIL, "nope nope nope", cfg, throttle) == "invalid"
    assert throttle.seconds_locked() > 0
    assert attempt_login(EMAIL, PASSWORD, cfg, throttle) == "locked"  # bypassing the disabled button gains nothing
    throttle2 = LoginThrottle()
    assert attempt_login(EMAIL, PASSWORD, cfg, throttle2) == "ok"


def test_expired_session_requires_login_again(make_app):
    at = make_app()
    at.session_state["auth"] = new_session(EMAIL, now=1)  # ancient
    at.run()
    assert [t.label for t in at.text_input] == ["Email", "Password"] and len(at.tabs) == 0


def test_valid_existing_session_skips_login(make_app):
    at = make_app()
    at.session_state["auth"] = new_session(EMAIL)
    at.run()
    assert [t.label for t in at.tabs][:2] == ["Chat", "Portfolio"]
    assert "Email" not in [t.label for t in at.text_input]


@pytest.mark.parametrize("settings", [
    Settings(secrets=Secrets()),                                   # nothing configured
    settings_with(email=None),                                     # email missing
    settings_with(password_hash=None),                             # hash missing
    settings_with(password_hash="not-a-valid-hash"),               # malformed hash
])
def test_misconfigured_auth_fails_closed(make_app, settings):
    at = make_app(settings).run()
    assert not at.exception
    assert any("This app is locked" in e.value for e in at.error)
    assert len(at.text_input) == 0 and len(at.tabs) == 0 and not at.chat_input


# ---- reading the credentials from st.secrets (the Streamlit Cloud path) ----------------------
@pytest.fixture
def cloud_like(monkeypatch, tmp_path):
    """No .env file and no AUTH env vars: credentials can only come from st.secrets, as on Cloud."""
    from src.core import settings as settings_module
    monkeypatch.setattr(settings_module, "ROOT_DIR", tmp_path)
    for name in ("AUTH_EMAIL", "AUTH_PASSWORD_HASH", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings_module.get_settings.cache_clear()
    st.cache_resource.clear()
    yield
    settings_module.get_settings.cache_clear()


def _cloud_app(monkeypatch, secrets):
    store = VectorStore.build(chunk_articles(ARTICLES, 400, 50), HashEmbedder())
    services = Services(FakeWorkflow(), MarketDataService(MockProvider()), NewsService(SampleProvider()),
                        Retriever(store, 3, 0.1), get_settings())
    monkeypatch.setattr(runtime, "create_runtime", lambda s=None: runtime.Runtime(services, runtime.AsyncRunner()))
    monkeypatch.setattr(login, "FAILURE_DELAY_SECONDS", 0)
    at = AppTest.from_file(APP, default_timeout=30)
    for k, v in secrets.items():
        at.secrets[k] = v
    return at


def test_credentials_in_st_secrets_unlock_the_login_form(cloud_like, monkeypatch):
    at = _cloud_app(monkeypatch, {"AUTH_EMAIL": EMAIL, "AUTH_PASSWORD_HASH": HASH}).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]  # not locked
    assert [t.label for t in at.text_input] == ["Email", "Password"]
    at = sign_in(at)
    assert "auth" in at.session_state and [t.label for t in at.tabs][:2] == ["Chat", "Portfolio"]


def test_missing_st_secrets_keeps_the_app_locked(cloud_like, monkeypatch):
    at = _cloud_app(monkeypatch, {"OPENAI_API_KEY": "sk-x"}).run()
    assert any("Login is not configured" in e.value for e in at.error) and len(at.text_input) == 0


def test_credentials_are_read_fresh_not_from_the_settings_cache(cloud_like, monkeypatch):
    """Regression: an early get_settings() call (before secrets existed) must not lock the login forever."""
    from src.core import settings as settings_module
    assert settings_module.get_settings().secrets.auth_email is None  # cache now holds 'no credentials'
    at = _cloud_app(monkeypatch, {"AUTH_EMAIL": EMAIL, "AUTH_PASSWORD_HASH": HASH}).run()
    assert [t.label for t in at.text_input] == ["Email", "Password"]


def test_locked_screen_names_only_the_missing_setting(cloud_like, monkeypatch):
    at = _cloud_app(monkeypatch, {"AUTH_EMAIL": EMAIL, "OPENAI_API_KEY": "sk-secret-name-check"}).run()
    msg = at.error[0].value
    assert "AUTH_PASSWORD_HASH not found" in msg and "AUTH_EMAIL not found" not in msg
    assert "OPENAI_API_KEY" not in msg and "sk-secret" not in msg and EMAIL not in msg


@pytest.mark.parametrize("secrets", [
    {"auth_email": EMAIL, "auth_password_hash": HASH},                         # lowercase names
    {"Auth_Email": EMAIL, "AUTH_PASSWORD_HASH": HASH},                         # mixed case
    {"general": {"AUTH_EMAIL": EMAIL, "AUTH_PASSWORD_HASH": HASH}},            # pasted under a [section]
    {"AUTH_EMAIL": f"  {EMAIL}  ", "AUTH_PASSWORD_HASH": f" {HASH} "},          # stray whitespace
])
def test_common_secrets_paste_mistakes_still_work(cloud_like, monkeypatch, secrets):
    at = _cloud_app(monkeypatch, secrets).run()
    assert not at.error, [e.value for e in at.error]
    assert "auth" in sign_in(at).session_state


def test_locked_screen_distinguishes_no_secrets_from_wrong_names(cloud_like, monkeypatch):
    none = _cloud_app(monkeypatch, {}).run()
    assert "No secrets are configured for this app at all" in none.error[0].value
    wrong = _cloud_app(monkeypatch, {"OPENAI_API_KEY": "sk-x"}).run()
    msg = wrong.error[0].value
    assert "Secrets exist for this app, but none of them is named" in msg
    assert "OPENAI_API_KEY" not in msg and "sk-x" not in msg  # still no key names or values
