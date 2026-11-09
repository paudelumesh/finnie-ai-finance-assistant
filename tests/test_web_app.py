"""Headless Streamlit tests (AppTest) with fake services: no network, no API keys."""
from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.core.settings import get_settings
from src.data.market_data import MarketDataService, MockProvider
from src.data.news import NewsService, SampleProvider
from src.rag.chunker import chunk_articles
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.core.auth import new_session
from src.web_app import runtime
from src.workflow.factory import Services
from src.workflow.graph import ChatResult
from tests.test_rag import ARTICLES

APP = str(Path(__file__).resolve().parents[1] / "src" / "web_app" / "app.py")


class FakeWorkflow:
    def __init__(self):
        self.calls = []

    async def chat(self, query, thread_id, profile, context):
        self.calls.append((query, thread_id, profile, context))
        return ChatResult(
            content=f"Echo: {query}", sources=[{"title": "Compound interest", "url": "https://x.io", "category": "basics"}],
            agents=["finance_qa"], route={}, guardrail={"flagged": False, "reasons": []})


@pytest.fixture
def app(monkeypatch):
    import streamlit as st
    st.cache_resource.clear()
    store = VectorStore.build(chunk_articles(ARTICLES, 400, 50), HashEmbedder())
    services = Services(
        workflow=FakeWorkflow(),
        market=MarketDataService(MockProvider()),
        news=NewsService(SampleProvider()),
        retriever=Retriever(store, 3, 0.1),
        settings=get_settings(),
    )
    monkeypatch.setattr(runtime, "create_runtime", lambda s=None: runtime.Runtime(services, runtime.AsyncRunner()))
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["auth"] = new_session("owner@example.com")
    at.services = services
    return at.run()


def test_app_starts_with_five_tabs_banner_and_no_errors(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Chat", "Portfolio", "Markets", "Goals", "Knowledge"]
    assert any("educational information only" in c.value for c in app.caption)
    assert app.title[0].value.endswith("Finnie")


def test_chat_roundtrip_and_context_passing(app):
    app.chat_input[0].set_value("what is compound interest?").run()
    assert not app.exception
    texts = [m.markdown[0].value for m in app.chat_message]
    assert texts[0] == "what is compound interest?" and texts[1].startswith("Echo: what is compound")
    query, thread, profile, ctx = app.services.workflow.calls[0]
    assert profile == {"risk_tolerance": "moderate", "knowledge_level": "beginner"} and thread


def test_example_button_sends_prompt(app):
    app.button(key="ex0").click().run()
    assert app.services.workflow.calls[0][0].startswith("What's the difference between an ETF")


def test_sidebar_profile_flows_to_workflow(app):
    app.sidebar.selectbox[0].select("high").run()
    app.chat_input[0].set_value("hi").run()
    assert app.services.workflow.calls[0][2]["risk_tolerance"] == "high"


def test_new_conversation_clears_history(app):
    app.chat_input[0].set_value("hi").run()
    first_thread = app.session_state.thread_id
    next(b for b in app.sidebar.button if "New conversation" in b.label).click().run()
    assert app.session_state.chat == [] and app.session_state.thread_id != first_thread


def test_portfolio_sample_analysis_then_explain(app):
    next(b for b in app.button if b.label == "Analyze portfolio").click().run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert {"Total value", "Risk rating", "Diversification", "Est. annual fees"} <= set(labels)
    assert app.session_state.analysis.total_value > 0 and len(app.session_state.holdings) == 4
    assert any("Sample data" in c.value for c in app.caption)  # MockProvider => labelled sample
    next(b for b in app.button if "Explain my portfolio" in b.label).click().run()
    query, _, _, ctx = app.services.workflow.calls[-1]
    assert "portfolio" in query.lower() and len(ctx["portfolio"]) == 4


def test_goals_plan_metrics_and_explain(app):
    next(b for b in app.button if b.label == "Build my plan").click().run()
    assert not app.exception and app.session_state.plan.on_track
    assert "Monthly saving used" in [m.label for m in app.metric]
    next(b for b in app.button if "Explain this plan" in b.label).click().run()
    ctx = app.services.workflow.calls[-1][3]
    assert ctx["goal"]["target"] == 60000 and ctx["goal"]["years"] == 8


def test_goals_validation_error_is_friendly(app):
    app.number_input[1].set_value(60.0)  # valid
    app.number_input[0].set_value(100.0)
    next(b for b in app.button if b.label == "Build my plan").click().run()
    assert not app.exception


def test_markets_overview_and_lookup(app):
    assert not app.exception
    assert any("S&P 500" in m.label for m in app.metric)
    app.text_input[0].set_value("aapl").run()
    assert not app.exception and "AAPL" in [m.label for m in app.metric]
    app.text_input[0].set_value("zzzz").run()
    assert any("No market data found for ZZZZ" in e.value for e in app.error)


def test_knowledge_search_and_browse(app):
    assert any("articles" in c.value for c in app.caption)
    app.text_input[-1].set_value("compound interest earns interest").run()
    assert not app.exception
    assert any("Compound interest" in m.value for m in app.markdown)
    next(s for s in app.selectbox if s.label == "Category").select("funds").run()
    assert not app.exception


def test_about_menu_items_open_without_errors(app):
    from src.web_app.about_menu import MENU
    assert list(MENU) == ["🏗️ Architecture", "🧭 High-level design",
                          "🔬 Technical architecture in detail", "🧰 Technologies used"]
    for label in MENU:
        next(b for b in app.button if b.key == f"about_{label}").click().run()
        assert not app.exception, label


def test_about_menu_is_top_right_not_in_the_sidebar(app):
    assert not [b for b in app.sidebar.button if b.key and b.key.startswith("about_")]
    assert [b for b in app.button if b.key and b.key.startswith("about_")]  # but it is on the page
    css = [m.value for m in app.markdown if "st-key-about_menu" in m.value]
    assert css and "position: fixed" in css[0] and "--finnie-about-right" in css[0]


def test_choosing_an_item_opens_its_dialog_and_closes_the_menu(app, monkeypatch):
    import streamlit as st
    from src.web_app import about_menu
    label = "🏗️ Architecture"
    monkeypatch.setitem(about_menu.MENU, label, lambda: st.markdown("DIALOG-OPENED:architecture"))
    assert app.session_state.about_nonce == 0
    before = next(b for b in app.button if b.key == f"about_{label}")
    before.click().run()
    assert [m.value for m in app.markdown].count("DIALOG-OPENED:architecture") == 1  # dialog opened once
    assert app.session_state.about_nonce == 1                                        # popover key changed => closed
    assert "about_pending" not in app.session_state                                  # consumed, won't reopen
    app.run()  # any later interaction must not reopen the dialog
    assert "DIALOG-OPENED:architecture" not in [m.value for m in app.markdown]


def test_menu_items_each_dispatch_their_own_dialog(app, monkeypatch):
    import streamlit as st
    from src.web_app import about_menu
    for label in list(about_menu.MENU):
        monkeypatch.setitem(about_menu.MENU, label, lambda l=label: st.markdown(f"OPEN:{l}"))
    for label in list(about_menu.MENU):
        next(b for b in app.button if b.key == f"about_{label}").click().run()
        opened = [m.value for m in app.markdown if m.value.startswith("OPEN:")]
        assert opened == [f"OPEN:{label}"], (label, opened)


def test_diagrams_are_valid_dot_and_content_is_consistent():
    from src.web_app import about_content as c
    for dot in (c.SYSTEM_DIAGRAM, c.FLOW_DIAGRAM, c.GRAPH_DIAGRAM, c.RESILIENCE_DIAGRAM, c.RAG_DIAGRAM):
        assert dot.startswith("digraph G {") and dot.count("{") == dot.count("}")
    assert len(c.AGENTS) == 6 and len(c.TECHNOLOGIES) >= 10
    for node in ("agent_finance_qa", "agent_portfolio_analysis", "agent_market_analysis",
                 "agent_goal_planning", "agent_tax_education", "agent_news_synthesizer", "guardrail"):
        assert node in c.GRAPH_DIAGRAM


def test_measure_script_uses_st_iframe_when_available_else_components(monkeypatch):
    import streamlit as st
    import streamlit.components.v1 as components
    from src.web_app import about_menu
    calls = []
    monkeypatch.setattr(st, "iframe", lambda src, **kw: calls.append(("iframe", kw)), raising=False)
    monkeypatch.setattr(components, "html", lambda src, **kw: calls.append(("components", kw)))
    about_menu._embed_measure_script()
    monkeypatch.delattr(st, "iframe", raising=False)
    about_menu._embed_measure_script()
    assert [c[0] for c in calls] == ["iframe", "components"]


def test_measure_script_failure_never_breaks_the_menu(monkeypatch):
    import streamlit as st
    from src.web_app import about_menu

    def boom(*a, **k):
        raise RuntimeError("iframe unavailable")

    monkeypatch.setattr(st, "iframe", boom, raising=False)
    about_menu._embed_measure_script()  # must not raise


def test_menu_placement_measures_real_header_buttons_not_the_toolbar_container():
    """Regression: the toolbar container spans the whole header, so measuring its edge pushed the menu
    to the far left. The script must measure the leftmost real button in the header's right half, and
    work at every width (a narrow-screen shortcut made the menu overlap the buttons)."""
    from src.web_app import about_menu
    js, css = about_menu._MEASURE_JS, about_menu._MENU_CSS
    assert '[data-testid="stHeader"] button' in js and '[data-testid="stHeader"] a' in js
    assert "r.left > win.innerWidth / 2" in js           # ignore the sidebar toggle on the left
    assert "getBoundingClientRect" in js and "stToolbar\"]')" not in js.split("forEach")[0]  # no container query
    assert "innerWidth <= 640" not in js and "max-width" not in css   # no narrow-screen special case
    assert "top: 16px" in css                             # vertically centred on the 28px header buttons
