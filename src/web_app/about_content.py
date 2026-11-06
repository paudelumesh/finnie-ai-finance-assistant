"""Content for the 'About Finnie' modals: Graphviz diagrams plus the text that explains them.

Everything here describes what the code actually does (settings come from config.yaml).
"""
from __future__ import annotations

_HEAD = """
  rankdir={rankdir}; bgcolor="transparent"; pad=0.2; nodesep=0.35; ranksep=0.45;
  fontname="Helvetica"; fontsize=12;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fontcolor="#0b0b0b",
        color="#8a8a85", fillcolor="#f0efec"];
  edge [color="#8a8a85", fontname="Helvetica", fontsize=10, fontcolor="#8a8a85"];
"""
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#cde2fb", "#f9d2c1", "#c3ecdc", "#fbe5b0", "#f0efec"


def _dot(body: str, rankdir: str = "TB") -> str:
    return "digraph G {" + _HEAD.format(rankdir=rankdir) + body + "}"


SYSTEM_DIAGRAM = _dot(f"""
  subgraph cluster_ui {{ label="Presentation · Streamlit"; style="rounded"; color="#8a8a85"; fontcolor="#8a8a85";
    ui [label="Chat · Portfolio · Markets · Goals · Knowledge · About", fillcolor="{BLUE}"]; }}
  subgraph cluster_orch {{ label="Orchestration · LangGraph"; style="rounded"; color="#8a8a85"; fontcolor="#8a8a85";
    router [label="Router\\n(LLM intent classifier + keyword fallback)", fillcolor="{ORANGE}"];
    agents [label="6 specialist agents\\n(run in parallel)", fillcolor="{ORANGE}"];
    synth [label="Synthesizer\\n(merge answers)", fillcolor="{ORANGE}"];
    guard [label="Guardrail\\n(education, not advice)", fillcolor="{ORANGE}"]; }}
  subgraph cluster_svc {{ label="Services"; style="rounded"; color="#8a8a85"; fontcolor="#8a8a85";
    rag [label="RAG retriever\\nFAISS + MiniLM", fillcolor="{AQUA}"];
    market [label="Market data\\ncache · retry · fallback", fillcolor="{AQUA}"];
    news [label="News search\\ncache · retry · fallback", fillcolor="{AQUA}"];
    calc [label="Calculators\\nportfolio · goals (pure code)", fillcolor="{AQUA}"]; }}
  subgraph cluster_ext {{ label="External"; style="rounded"; color="#8a8a85"; fontcolor="#8a8a85";
    openai [label="OpenAI API\\n(LLM)", fillcolor="{YELLOW}"];
    yf [label="yFinance\\n(quotes, history)", fillcolor="{YELLOW}"];
    tav [label="Tavily\\n(news search)", fillcolor="{YELLOW}"]; }}
  ui -> router; router -> agents; agents -> synth; synth -> guard; guard -> ui [label="answer + sources"];
  agents -> rag; agents -> market; agents -> news; agents -> calc;
  market -> yf; news -> tav; router -> openai [style=dashed]; agents -> openai [style=dashed];
""")

FLOW_DIAGRAM = _dot(f"""
  q [label="1. User message\\n(+ profile, portfolio, goal form)", fillcolor="{BLUE}"];
  r [label="2. Router picks 1+ intents\\n(uses recent chat history)", fillcolor="{ORANGE}"];
  a [label="3. Agents run in parallel\\n(each: gather data → compute → explain)", fillcolor="{ORANGE}"];
  s [label="4. Synthesizer merges answers\\n(one agent: pass through)", fillcolor="{ORANGE}"];
  g [label="5. Guardrail\\n(advice check + disclaimer)", fillcolor="{ORANGE}"];
  u [label="6. UI shows answer,\\nagents used and sources", fillcolor="{BLUE}"];
  q -> r -> a -> s -> g -> u;
""", "LR")

GRAPH_DIAGRAM = _dot(f"""
  START [shape=circle, label="START", fillcolor="{GRAY}"]; END [shape=circle, label="END", fillcolor="{GRAY}"];
  router [label="router\\nclassify + resolve", fillcolor="{ORANGE}"];
  refuse [label="refuse\\noff-topic / unavailable", fillcolor="{YELLOW}"];
  qa [label="agent_finance_qa", fillcolor="{AQUA}"]; pf [label="agent_portfolio_analysis", fillcolor="{AQUA}"];
  mk [label="agent_market_analysis", fillcolor="{AQUA}"]; gl [label="agent_goal_planning", fillcolor="{AQUA}"];
  tx [label="agent_tax_education", fillcolor="{AQUA}"]; nw [label="agent_news_synthesizer", fillcolor="{AQUA}"];
  synth [label="synthesize\\nmerge (LLM) or pass through", fillcolor="{ORANGE}"];
  guard [label="guardrail\\nrewrite advice · add disclaimer\\nwrite chat memory", fillcolor="{ORANGE}"];
  START -> router;
  router -> qa [label="Send"]; router -> pf [label="Send"]; router -> mk [label="Send"];
  router -> gl [label="Send"]; router -> tx [label="Send"]; router -> nw [label="Send"];
  router -> refuse [style=dashed];
  qa -> synth; pf -> synth; mk -> synth; gl -> synth; tx -> synth; nw -> synth; refuse -> synth;
  synth -> guard -> END;
""")

RESILIENCE_DIAGRAM = _dot(f"""
  req [label="Request\\n(quote / history / news)", fillcolor="{BLUE}"];
  cache [label="TTL cache\\n30 min", fillcolor="{AQUA}"];
  prov [label="Live provider\\nyFinance / Tavily\\nretry: wait 1s, 2s", fillcolor="{AQUA}"];
  stale [label="Stale cache\\nlast known value", fillcolor="{YELLOW}"];
  fb [label="Fallback data\\nsample prices / sample news\\n(known symbols only)", fillcolor="{YELLOW}"];
  err [label="Clear error\\n(never invented data)", fillcolor="{ORANGE}"];
  req -> cache [label="hit → 'cache'"]; cache -> prov [label="miss / expired"];
  prov -> req [label="ok → 'live'", style=dashed];
  prov -> stale [label="fails"]; stale -> req [label="has value → 'stale'", style=dashed];
  stale -> fb [label="none"]; fb -> req [label="→ 'mock' / 'sample'", style=dashed];
  fb -> err [label="unknown symbol", style=dotted];
""", "LR")

RAG_DIAGRAM = _dot(f"""
  subgraph cluster_off {{ label="Offline · build time"; style="rounded"; color="#8a8a85"; fontcolor="#8a8a85";
    t [label="topics.yaml\\n103 topics · 12 categories", fillcolor="{GRAY}"];
    d [label="LLM drafts articles\\n(reviewed: false until checked)", fillcolor="{YELLOW}"];
    md [label="Markdown + front-matter\\ntitle · category · source", fillcolor="{GRAY}"];
    ch [label="Chunker\\n800 chars, 120 overlap\\nsection-aware", fillcolor="{AQUA}"];
    em [label="Embeddings\\nall-MiniLM-L6-v2 (384-d)", fillcolor="{AQUA}"];
    ix [label="FAISS IndexFlatIP\\n(cosine on unit vectors)", fillcolor="{AQUA}"];
    t -> d -> md -> ch -> em -> ix; }}
  subgraph cluster_on {{ label="Online · per question"; style="rounded"; color="#8a8a85"; fontcolor="#8a8a85";
    qq [label="Question", fillcolor="{BLUE}"];
    sr [label="Embed + search\\nall chunks scored", fillcolor="{ORANGE}"];
    fl [label="Category: strict filter (UI)\\nor soft hint (router)", fillcolor="{ORANGE}"];
    th [label="min_score 0.25\\ntop-k 4", fillcolor="{ORANGE}"];
    pr [label="Prompt with numbered\\npassages [1] [2]…", fillcolor="{ORANGE}"];
    an [label="Cited answer\\n+ source list", fillcolor="{BLUE}"];
    qq -> sr -> fl -> th -> pr -> an; }}
  ix -> sr [style=dashed, label="loaded at startup"];
""")

AGENTS = [
    ("Finance Q&A", "Explains concepts (ETFs, compound interest…)", "RAG over the knowledge base", "Cites passages; says so when nothing matches"),
    ("Portfolio Analysis", "Reviews your holdings", "Holdings + live prices", "Value, allocation, diversification score, fees, risk, warnings are computed in code"),
    ("Market Analysis", "Quotes and recent trend", "yFinance via the market layer", "Prices and % changes come from data; the LLM only explains"),
    ("Goal Planning", "How much to save to hit a target", "Your goal + risk profile", "Future-value maths in code; three illustrative return scenarios"),
    ("Tax Education", "401(k), IRA, Roth, capital gains", "RAG, tax and retirement preferred", "Never says what you owe or which account to pick"),
    ("News Synthesizer", "Summarises current financial news", "Tavily search", "Only summarises the articles returned, with links"),
]

PRINCIPLES = [
    ("Code computes, the LLM explains", "Portfolio metrics and goal projections are deterministic functions, so numbers are reproducible and testable. The model is given the results and told not to recompute or invent any."),
    ("Education, not advice", "Every answer carries a disclaimer. A guardrail reframes 'should I buy X?' into factors to consider and rewrites any generated recommendation."),
    ("Grounded and attributable", "Concept answers use retrieved passages with inline citations [1], and sources are listed. News answers link to the original articles."),
    ("Degrade gracefully, never fabricate", "External calls are cached, retried and fall back to stale or sample data, always labelled (Live, Cached, Delayed, Sample). Unknown tickers return an error instead of a made-up price."),
    ("One failure never sinks the answer", "Agents are isolated: errors and timeouts become a friendly message while the other agents' answers still go through."),
]

TECHNOLOGIES = [
    ("Orchestration", "LangGraph 1.2", "StateGraph with parallel fan-out (Send), conditional routing and an in-memory checkpointer for per-conversation memory"),
    ("LLM interface", "LangChain (langchain-core, langchain-openai)", "Async chat model wrapper behind our own small LLMClient interface, so providers and fakes are swappable"),
    ("Language model", "OpenAI gpt-4o-mini", "Router classification, goal-parameter extraction, explanations, answer merging. Model name lives in config.yaml"),
    ("Embeddings", "sentence-transformers 6.x · all-MiniLM-L6-v2", "384-dimension sentence embeddings, run locally (no API cost)"),
    ("Vector store", "FAISS (faiss-cpu 1.15)", "Exact cosine search (IndexFlatIP) over about 700 chunks; persisted to disk"),
    ("Market data", "yfinance 1.8", "Quotes and price history, run in worker threads so the async app never blocks"),
    ("News search", "Tavily (tavily-python 0.8)", "News-topic search with dates and source URLs"),
    ("Web UI", "Streamlit 1.67 · Plotly 7.1 · pandas 3.1", "Five-tab app, interactive charts, Graphviz diagrams; a dedicated async runner bridges Streamlit and the async workflow"),
    ("Data and config", "Pydantic 2 · PyYAML · python-dotenv", "Typed settings from config.yaml, API keys from .env (or st.secrets on Streamlit Cloud)"),
    ("Testing", "pytest · pytest-asyncio · pytest-cov · Streamlit AppTest", "Mocked LLM and APIs, headless UI tests, coverage reporting"),
    ("Access control", "Python stdlib: hashlib.scrypt, hmac", "Single-owner login with a salted password hash (no plaintext stored), lockout and session expiry; no sign-up so a public deployment can't spend your model budget"),
    ("Integration", "Model Context Protocol (mcp)", "Exposes Finnie's deterministic tools to Claude Desktop"),
    ("Language", "Python 3.12 · asyncio", "Async throughout; blocking libraries are wrapped with asyncio.to_thread"),
]
