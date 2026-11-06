# Finnie: multi-agent finance assistant. Build once; the image contains the FAISS index and the
# embedding model, so first start is fast and needs no extra downloads (only API calls at runtime).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    PYTHONPATH=/app

WORKDIR /app

# CPU-only torch keeps the image small; then the rest of the dependencies.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .

# Bake the vector index and embedding model into the image.
RUN python -m scripts.build_index

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').read()==b'ok' else 1)"

# API keys are NOT in the image: pass them with --env-file .env (see README).
CMD ["streamlit", "run", "src/web_app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
