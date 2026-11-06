# Finnie – AI Finance Assistant

Multi-agent financial education assistant (LangGraph + RAG + Streamlit). Educational only, not financial advice.

## Setup
```bash
/opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your keys
pytest --cov=src
```
## Run
```bash
python -m scripts.build_index        # once: builds the FAISS index from src/data/articles
streamlit run src/web_app/app.py       # web UI
python -m scripts.chat_cli             # terminal chat
```
Demo walkthrough: `DEMO.md`.

See `PLAN.md` for architecture decisions and phases. Full docs land in Phase 5.
