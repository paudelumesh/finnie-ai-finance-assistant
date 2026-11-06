"""Knowledge tab: search the RAG corpus (with category filter) and browse articles."""
from __future__ import annotations

import streamlit as st

from src.rag.loader import Article, load_articles
from src.web_app.runtime import Runtime

ALL = "All categories"


@st.cache_data(show_spinner=False)
def _articles() -> list[Article]:
    return load_articles()


def render(rt: Runtime) -> None:
    retriever = rt.services.retriever
    cats = [ALL] + retriever.store.categories
    c1, c2 = st.columns([1, 3])
    category = c1.selectbox("Category", cats)
    query = c2.text_input("Search the knowledge base", placeholder="e.g. how do dividends work?")

    if query:
        hits = retriever.retrieve(query, None if category == ALL else category, strict=True).hits
        if not hits:
            st.info("No matching passages. Try different words or another category.")
        for h in hits:
            with st.container(border=True):
                st.markdown(f"**{h.chunk.title}** · _{h.chunk.category}_ · relevance {h.score:.2f}")
                st.write(h.chunk.text)
                if h.chunk.source_url:
                    st.caption(f"Reference: [{h.chunk.source}]({h.chunk.source_url})")
        return

    articles = [a for a in _articles() if category == ALL or a.category == category]
    st.caption(f"{len(articles)} articles. Content is educational and AI-drafted; the ✅ mark means a human reviewed it.")
    for a in sorted(articles, key=lambda a: (a.category, a.title)):
        with st.expander(f"{'✅ ' if a.reviewed else ''}{a.title}  ·  {a.category}"):
            st.markdown(a.body)
            if a.source_url:
                st.caption(f"Reference: [{a.source}]({a.source_url})")
