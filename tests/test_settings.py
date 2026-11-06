from src.core.settings import load_settings


def test_defaults_load_from_config_yaml(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = load_settings()
    assert s.llm.provider == "openai"
    assert s.market_data.cache_ttl_seconds == 1800
    assert s.rag.top_k == 4


def test_secret_read_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert load_settings().secrets.openai_api_key == "sk-test"


def test_missing_config_file_uses_defaults(tmp_path):
    s = load_settings(tmp_path / "nope.yaml")
    assert s.llm.model == "gpt-4o-mini"
