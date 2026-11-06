# Finnie – AI Finance Assistant

Multi-agent financial education assistant (LangGraph + RAG + Streamlit). Educational only, not financial advice.

## Setup
```bash
/opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your keys
pytest --cov=src
```
See `PLAN.md` for architecture decisions and phases. Full docs land in Phase 5.
