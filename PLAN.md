# Finnie – Build Plan (agreed decisions)

Deadline: ~Nov 22, 2026 (4 weeks), ~15–20 h/week. Python 3.12 venv.

## Decisions
| Area | Decision |
|---|---|
| LLM | OpenAI (program key) primary; model name in `config.yaml`, Gemini Flash as swap |
| Orchestration | LangGraph StateGraph, async throughout (`ainvoke`, `asyncio.gather`) |
| Agents | 6; build order Q&A → Portfolio → Market → Goal → Tax → News (News cut first if slipping) |
| Router | LLM classifier (structured intent + confidence), keyword fallback; multi-intent fans out, synthesizer merges |
| Memory | LangGraph checkpointer (in-memory) + user profile (risk, goals, level) |
| KB | 60 → 100+ LLM-drafted Markdown articles, spot-checked vs Investopedia/Bogleheads/investor.gov; front-matter: category, source URL |
| RAG | FAISS + all-MiniLM-L6-v2, category filter, source attribution |
| Market data | yFinance primary, Alpha Vantage optional; 30-min TTL cache; mock fallback; freshness indicator |
| News | Tavily; cache + bundled sample headlines fallback |
| Portfolio | CSV upload + manual entry; value, allocation (class/sector), HHI diversification score, expense ratios (bundled ETF table), risk rating. Code computes, LLM explains |
| Goals | Deterministic FV math, risk-profile return ranges, conservative/expected/optimistic scenarios |
| UI | Streamlit, 5 tabs (Chat, Portfolio, Markets, Goals, Knowledge), Plotly, disclaimer banner |
| Guardrails | Output-check node: disclaimer + rewrite "should I buy X" into educational framing |
| Tests | pytest, mocked LLM/APIs, 30+ labelled router queries, ≥80% coverage, written per phase |
| Secrets | non-secret → `config.yaml`; keys → git-ignored `.env` (+ `.env.example`); `st.secrets` compatible |
| Stretch | MCP server, Docker, Streamlit Cloud – only after Phase 4 is done |

## Phases (approval gate after each; commit only when asked)
1. Foundation – skeleton, settings, base agent, LLM wrapper, logging
2. Knowledge & RAG – corpus, chunking, FAISS, retrieval + attribution, Q&A agent
3. Data agents – market layer (cache/fallback), Portfolio, Goal, router + LangGraph workflow (biggest risk)
4. UI & remaining agents – Streamlit tabs, Tax, News, guardrail node
5. Hardening & delivery – tests ≥80%, README + design doc + diagrams, DEMO.md/video, stretch goals

## Demo scenes (keep DEMO.md updated)
1. Multi-turn Q&A flowing into a goal plan
2. Portfolio CSV upload + analysis
3. Live market lookup with freshness indicator
4. Multi-intent query showing agent fan-out
