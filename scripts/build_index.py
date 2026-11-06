"""Build the FAISS index from the article corpus.  Usage: python -m scripts.build_index"""
from __future__ import annotations

from collections import Counter

from src.core.settings import get_settings
from src.rag.chunker import chunk_articles
from src.rag.embeddings import SentenceTransformerEmbedder
from src.rag.loader import load_articles
from src.rag.vector_store import DEFAULT_INDEX_DIR, VectorStore


def main() -> None:
    cfg = get_settings().rag
    articles = load_articles()
    chunks = chunk_articles(articles, cfg.chunk_size, cfg.chunk_overlap)
    print(f"{len(articles)} articles -> {len(chunks)} chunks")
    print("by category:", dict(Counter(a.category for a in articles)))
    store = VectorStore.build(chunks, SentenceTransformerEmbedder(cfg.embedding_model))
    store.save()
    print(f"index saved to {DEFAULT_INDEX_DIR}")


if __name__ == "__main__":
    main()
