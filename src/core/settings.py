"""Typed settings: non-secret values from config.yaml, API keys from env / .env."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


class LLMSettings(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    timeout_seconds: int = 30
    max_retries: int = 3


class MarketDataSettings(BaseModel):
    provider: str = "yfinance"
    cache_ttl_seconds: int = 1800
    use_mock_on_failure: bool = True


class NewsSettings(BaseModel):
    provider: str = "tavily"
    cache_ttl_seconds: int = 1800
    max_results: int = 5
    use_sample_on_failure: bool = True


class RAGSettings(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_score: float = 0.25


class RouterSettings(BaseModel):
    confidence_threshold: float = 0.6
    classifier_timeout_seconds: int = 15
    agent_timeout_seconds: int = 60


class LoggingSettings(BaseModel):
    level: str = "INFO"


class Secrets(BaseModel):
    openai_api_key: str | None = None
    google_api_key: str | None = None
    alpha_vantage_api_key: str | None = None
    tavily_api_key: str | None = None
    auth_email: str | None = None
    auth_password_hash: str | None = None  # salted scrypt hash, never the password


class Settings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)
    news: NewsSettings = Field(default_factory=NewsSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    secrets: Secrets = Field(default_factory=Secrets)


def _lookup_streamlit_secret(name: str) -> str | None:
    """Find `name` in st.secrets, tolerating the common paste mistakes: different capitalisation,
    or the keys sitting inside a [section] table (one level deep)."""
    import streamlit as st

    secrets, wanted = st.secrets, name.lower()
    tables = []
    for key in secrets:
        value = secrets[key]
        if str(key).lower() == wanted and isinstance(value, str) and value.strip():
            return value.strip()
        if hasattr(value, "keys"):
            tables.append(value)
    for table in tables:
        for key in table:
            value = table[key]
            if str(key).lower() == wanted and isinstance(value, str) and value.strip():
                return value.strip()
    return None


def secrets_available() -> bool:
    """True if Streamlit has any secrets configured at all (says nothing about their names)."""
    try:
        import streamlit as st

        return len(st.secrets) > 0
    except Exception:
        return False


def _read_secret(name: str) -> str | None:
    """Env var first (covers .env); fall back to st.secrets when running on Streamlit Cloud."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return _lookup_streamlit_secret(name)
    except Exception:  # no secrets file / not running under Streamlit
        return None


def load_auth_credentials() -> tuple[str | None, str | None]:
    """Read the login email and password hash fresh on every call (never from the settings cache).

    Settings are cached for the life of the process. If the cache were filled before the secrets
    were readable, a missing login would stay missing until a full restart.
    """
    load_dotenv(ROOT_DIR / ".env")
    return _read_secret("AUTH_EMAIL"), _read_secret("AUTH_PASSWORD_HASH")


def load_settings(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Settings:
    load_dotenv(ROOT_DIR / ".env")
    raw: dict[str, Any] = {}
    path = Path(config_path)
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
    raw["secrets"] = {
        "openai_api_key": _read_secret("OPENAI_API_KEY"),
        "google_api_key": _read_secret("GOOGLE_API_KEY"),
        "alpha_vantage_api_key": _read_secret("ALPHA_VANTAGE_API_KEY"),
        "tavily_api_key": _read_secret("TAVILY_API_KEY"),
        "auth_email": _read_secret("AUTH_EMAIL"),
        "auth_password_hash": _read_secret("AUTH_PASSWORD_HASH"),
    }
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
