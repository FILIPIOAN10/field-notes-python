"""Tiny RAG layer: chunk a document, embed chunks, retrieve the most relevant ones by cosine similarity.

Default embedder is a local hashed bag-of-words (no API key, deterministic). Set VOYAGE_API_KEY to use
Voyage AI embeddings (Anthropic's recommended embedding partner) instead — same interface.
"""
import hashlib
import os
import re

import numpy as np

_WORD = re.compile(r"[a-zA-ZÀ-ž0-9]+")


def chunk_text(text: str, max_words: int = 80, overlap: int = 20) -> list[str]:
    """Split into overlapping word windows so no sentence is lost at a boundary."""
    words = text.split()
    if not words:
        return []
    step = max(1, max_words - overlap)
    return [" ".join(words[i : i + max_words]) for i in range(0, max(1, len(words) - overlap), step)]


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, t in enumerate(texts):
            for w in _WORD.findall(t.lower()):
                if len(w) > 2:
                    out[row, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class VoyageEmbedder:
    def __init__(self, model: str = "voyage-3-lite"):
        import requests

        self.requests, self.model = requests, model

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self.requests.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}"},
            json={"input": texts, "model": self.model}, timeout=60,
        )
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        v = np.array([d["embedding"] for d in data], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_embedder():
    return VoyageEmbedder() if os.getenv("VOYAGE_API_KEY") else HashingEmbedder()


class VectorStore:
    """In-memory vector index (swap for pgvector / OpenSearch in production)."""

    def __init__(self, chunks: list[str], embedder=None):
        self.chunks = chunks
        self.embedder = embedder or make_embedder()
        self.vectors = self.embedder.embed(chunks) if chunks else np.zeros((0, 1))

    @classmethod
    def from_text(cls, text: str, embedder=None) -> "VectorStore":
        return cls(chunk_text(text), embedder)

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        if not self.chunks:
            return []
        scores = self.vectors @ self.embedder.embed([query])[0]
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def context_block(hits: list[tuple[str, float]]) -> str:
    return "\n".join(f"[{i}] {c}" for i, (c, _) in enumerate(hits, 1))
