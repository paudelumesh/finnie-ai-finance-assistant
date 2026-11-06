"""Create the login credentials (email + salted scrypt hash). The password itself is never saved.

Interactive (recommended; the password is not echoed or stored in shell history):
    python -m scripts.set_password --email you@example.com --write-env
Non-interactive (password on stdin):
    printf '%s' "$PASSWORD" | python -m scripts.set_password --email you@example.com --stdin --write-env

--write-env updates AUTH_EMAIL and AUTH_PASSWORD_HASH in the git-ignored .env.
Without it, the lines are printed so you can paste them into Streamlit Cloud > Settings > Secrets.
"""
from __future__ import annotations

import argparse
import getpass
import re
import sys

from src.core.auth import hash_password
from src.core.settings import ROOT_DIR

ENV_PATH = ROOT_DIR / ".env"


def update_env(text: str, values: dict[str, str]) -> str:
    """Replace existing KEY=... lines or append new ones, leaving everything else untouched."""
    lines = text.splitlines()
    for key, value in values.items():
        pattern = re.compile(rf"^{re.escape(key)}=")
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = f"{key}={value}"
                break
        else:
            lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--stdin", action="store_true", help="read the password from stdin")
    ap.add_argument("--write-env", action="store_true", help="write the values into .env")
    args = ap.parse_args()

    if args.stdin:
        password = sys.stdin.read().rstrip("\n")
    else:
        password = getpass.getpass("New password: ")
        if password != getpass.getpass("Repeat password: "):
            sys.exit("Passwords do not match.")
    try:
        hashed = hash_password(password)
    except ValueError as exc:
        sys.exit(str(exc))

    values = {"AUTH_EMAIL": args.email.strip().lower(), "AUTH_PASSWORD_HASH": hashed}
    if args.write_env:
        current = ENV_PATH.read_text() if ENV_PATH.exists() else ""
        ENV_PATH.write_text(update_env(current, values))
        print(f"Wrote AUTH_EMAIL and AUTH_PASSWORD_HASH to {ENV_PATH} (git-ignored).")
    else:
        for k, v in values.items():
            print(f'{k} = "{v}"')
    print("For Streamlit Cloud, add the same two keys under Settings > Secrets.")


if __name__ == "__main__":
    main()
