# Demo script (5–10 min)

Run: `streamlit run src/web_app/app.py` (needs `.env` keys and `python -m scripts.build_index` once).

1. **Multi-turn Q&A flowing into a goal plan** (Chat tab)
   - "What's the difference between an ETF and a mutual fund?" → cited answer, sources expander
   - "I want $60,000 in 8 years and I have $5,000. How much should I save monthly?"
   - "What about 12 years?" → follow-up resolved from memory
   - Goals tab: build the same plan, show the three-scenario chart
2. **Portfolio CSV upload + analysis** (Portfolio tab)
   - Upload `data_samples/sample_portfolio.csv` (or "Use sample portfolio") → Analyze
   - Point out: diversification score, concentration warning, no-bonds warning, fee estimate
   - "Explain my portfolio with Finnie"
3. **Live market lookup with freshness indicator** (Markets tab)
   - Overview + indexed comparison chart; look up `NVDA`; click Refresh; note the Live / Cached label
4. **Multi-intent fan-out + guardrail** (Chat tab)
   - "Should I buy NVDA? Also explain what an ETF is." → two agents, merged answer, advice reframed
   - "What's the latest news on interest rates?" → News agent with source links
   - "How are capital gains taxed?" → Tax agent
   - "Who won the world cup?" → polite refusal
