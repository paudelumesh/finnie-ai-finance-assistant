"""Write a ready-to-paste, validated TOML file of the secrets the app needs on Streamlit Cloud.

    python -m scripts.export_secrets          # reads .env, writes cloud_secrets.toml

Open cloud_secrets.toml, copy ALL of it into the app's Settings > Secrets box, save, reboot the app,
then DELETE the file (it contains your API keys; it is git-ignored). Nothing is printed to the terminal.
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

from dotenv import dotenv_values

from src.core.settings import ROOT_DIR

KEYS = ("OPENAI_API_KEY", "TAVILY_API_KEY", "AUTH_EMAIL", "AUTH_PASSWORD_HASH")
REQUIRED = ("OPENAI_API_KEY", "AUTH_EMAIL", "AUTH_PASSWORD_HASH")
OUT_PATH = ROOT_DIR / "cloud_secrets.toml"


def build_toml(values: dict[str, str | None]) -> str:
    """Quote every value as a TOML basic string (json.dumps escaping is valid TOML for these)."""
    lines = [f"{k} = {json.dumps(values[k].strip())}" for k in KEYS if values.get(k) and values[k].strip()]
    text = "\n".join(lines) + "\n"
    parsed = tomllib.loads(text)  # must parse back to exactly what we meant
    for k in KEYS:
        if values.get(k) and values[k].strip() and parsed.get(k) != values[k].strip():
            raise ValueError(f"{k} did not round-trip through TOML")
    return text


def main(env_path: Path = ROOT_DIR / ".env", out_path: Path = OUT_PATH) -> None:
    values = dotenv_values(env_path)
    missing = [k for k in REQUIRED if not (values.get(k) or "").strip()]
    if missing:
        sys.exit(f"Missing in {env_path.name}: {', '.join(missing)}. "
                 "Run `python -m scripts.set_password --email you@example.com --write-env` for the AUTH_ values.")
    out_path.write_text(build_toml(values), encoding="utf-8")
    out_path.chmod(0o600)
    found = [k for k in KEYS if (values.get(k) or "").strip()]
    print(f"Wrote {out_path.name} with {len(found)} secrets ({', '.join(found)}); values not shown.")
    print("Copy its whole contents into Streamlit Cloud > app > Settings > Secrets, save, reboot, then delete the file.")


if __name__ == "__main__":
    main()
