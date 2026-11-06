"""FAISS-backed vector store with category filtering and on-disk persistence."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss

from src.rag.chunker import Chunk
from src.rag.embeddings import Embedder

DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "index"


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], embedder: Embedder) -> None:
        self._index = index
        self._chunks = chunks
        self._embedder = embedder

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def categories(self) -> list[str]:
        return sorted({c.category for c in self._chunks})

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "VectorStore":
        if not chunks:
            raise ValueError("cannot build an index from zero chunks")
        vectors = embedder.embed([c.text for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine on unit vectors
        index.add(vectors)
        return cls(index, chunks, embedder)

    def search(
        self,
        query: str,
        k: int = 4,
        category: str | None = None,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        if not query.strip():
            return []
        vec = self._embedder.embed([query])
        # The corpus is small, so scan everything and filter; keeps category filtering exact.
        scores, ids = self._index.search(vec, len(self._chunks))
        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            if category and chunk.category != category:
                continue
            if score < min_score:
                break  # scores are sorted descending
            hits.append(SearchHit(chunk, float(score)))
            if len(hits) == k:
                break
        return hits

    def save(self, directory: Path = DEFAULT_INDEX_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        (directory / "chunks.json").write_text(json.dumps([asdict(c) for c in self._chunks]))

    @classmethod
    def load(cls, embedder: Embedder, directory: Path = DEFAULT_INDEX_DIR) -> "VectorStore":
        directory = Path(directory)
        index_path, chunks_path = directory / "index.faiss", directory / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No index in {directory}. Run `python -m scripts.build_index` first."
            )
        index = faiss.read_index(str(index_path))
        chunks = [Chunk(**c) for c in json.loads(chunks_path.read_text())]
        if index.ntotal != len(chunks):
            raise ValueError("index and chunk metadata are out of sync; rebuild the index")
        return cls(index, chunks, embedder)

