"""Login gate. Nothing else in the app (not even the model or services) starts until this passes."""
from __future__ import annotations

import time

import streamlit as st

from src.core.auth import (
    AuthConfig, AuthConfigError, LoginThrottle, check_credentials, new_session, session_valid,
)
from src.core.settings import load_auth_credentials

FAILURE_DELAY_SECONDS = 1.0  # slows down guessing from a single connection


@st.cache_resource
def _throttle() -> LoginThrottle:
    return LoginThrottle()


def _config() -> AuthConfig:
    return AuthConfig.from_values(*load_auth_credentials())


def attempt_login(email: str, password: str, cfg: AuthConfig, throttle: LoginThrottle) -> str:
    """Return 'ok', 'invalid' or 'locked'. The lock is checked FIRST and here, not in the UI, so a
    request that bypasses a disabled button still can't test passwords during a lockout."""
    if throttle.seconds_locked() > 0:
        return "locked"
    if check_credentials(email, password, cfg):
        throttle.record_success()
        return "ok"
    throttle.record_failure()
    return "invalid"


def logout() -> None:
    st.session_state.pop("auth", None)


def require_login() -> str | None:
    """Return the signed-in email, or render the login form and return None."""
    if session_valid(st.session_state.get("auth")):
        return st.session_state["auth"]["email"]
    st.session_state.pop("auth", None)  # expired or missing

    try:
        cfg = _config()
    except AuthConfigError as exc:  # fail closed: a misconfigured app is locked, never open
        st.title("💬 Finnie")
        st.error(f"This app is locked. {exc}")
        return None

    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.title("💬 Finnie")
        st.caption("Private. Please sign in.")
        throttle = _throttle()
        with st.form("login", clear_on_submit=False):
            email = st.text_input("Email", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch",
                                              disabled=throttle.seconds_locked() > 0)
        result = attempt_login(email, password, cfg, throttle) if submitted else None
        if result == "ok":
            st.session_state["auth"] = new_session(cfg.email)
            st.rerun()
        if result == "invalid":
            time.sleep(FAILURE_DELAY_SECONDS)  # slows guessing from one connection
        locked = throttle.seconds_locked()
        if locked:
            st.error(f"Too many failed attempts. Try again in {locked // 60 + 1} min.")
        elif result == "invalid":
            st.error("Invalid email or password.")  # same message whichever field was wrong
    return None
