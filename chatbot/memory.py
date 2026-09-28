"""In-memory knowledge base built from uploaded documents.

Text is split into overlapping chunks; each chunk is embedded with the
configured embedding model. Questions are embedded the same way and answered
with the most similar chunks (cosine similarity). If embeddings are
unavailable (no API key yet, or an embedding error), a lightweight keyword
scorer takes over so the chatbot still works offline.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, TOP_K
from .loader import RawDocument

# Some OpenAI-compatible providers (e.g. OpenRouter) have no embeddings
# endpoint. Set EMBEDDING_MODEL=off in .env to use keyword search only.
_EMBEDDINGS_OFF = EMBEDDING_MODEL.strip().lower() in {"", "off", "none", "disabled"}

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Errors that mean the provider genuinely has no embedding endpoint — only
# these permanently disable embeddings. Transient failures (timeouts, 429s,
# dropped connections) must not, or retrieval would silently degrade forever.
_PERMANENT_EMBED_ERRORS = ("404", "does not exist", "not supported", "unknown model", "invalid model", "no endpoint")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= size:
        return [text] if text else []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Try to break at a sentence/line boundary near the end of the chunk.
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [c for c in chunks if c]


@dataclass
class Chunk:
    text: str
    source: str
    embedding: Optional[List[float]] = None
    terms: List[str] = field(default_factory=list)


class KnowledgeBase:
    """Stores document chunks and retrieves the most relevant ones for a query."""

    def __init__(self, client=None, embedding_model: str = EMBEDDING_MODEL):
        self.client = client  # OpenAI-compatible client, may be None (offline mode)
        self.embedding_model = embedding_model
        self.embeddings_enabled = not _EMBEDDINGS_OFF and client is not None
        self.chunks: List[Chunk] = []
        self.sources: List[str] = []  # unique source labels, for the UI

    # -- ingestion ------------------------------------------------------------
    def add_documents(self, documents: List[RawDocument]) -> int:
        """Add extracted documents; returns the number of new chunks."""
        new_chunks: List[Chunk] = []
        for source, text in documents:
            for piece in _chunk_text(text):
                # Sheet/page names ("Q1 Sales", "page 3") are only in the source
                # label — index them too so "Q1" can rank the right sheet.
                terms = _tokenize(piece) + _tokenize(source)
                chunk = Chunk(text=piece, source=source, terms=terms)
                new_chunks.append(chunk)
        self._embed(new_chunks)
        self.chunks.extend(new_chunks)
        for chunk in new_chunks:
            if chunk.source not in self.sources:
                self.sources.append(chunk.source)
        return len(new_chunks)

    def clear(self) -> None:
        self.chunks.clear()
        self.sources.clear()

    def _embed(self, chunks: List[Chunk]) -> None:
        if not self.embeddings_enabled:
            return
        texts = [c.text for c in chunks]
        try:
            # Batch to keep requests small.
            for i in range(0, len(texts), 100):
                batch = texts[i : i + 100]
                resp = self.client.embeddings.create(model=self.embedding_model, input=batch)
                for chunk, item in zip(chunks[i : i + 100], resp.data):
                    chunk.embedding = item.embedding
        except Exception as exc:
            # Embeddings are an enhancement; keyword search still works. Only a
            # permanent provider-side refusal disables them for good — a flaky
            # network or rate limit should not degrade every future search.
            if any(marker in str(exc).lower() for marker in _PERMANENT_EMBED_ERRORS):
                self.embeddings_enabled = False
            for chunk in chunks:
                chunk.embedding = None

    # -- retrieval -------------------------------------------------------------
    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, str, float]]:
        """Return ``(source, text, score)`` for the most relevant chunks."""
        if not self.chunks:
            return []
        query = query.strip()
        if not query:
            return []
        if self.embeddings_enabled:
            try:
                return self._embedding_search(query, top_k)
            except Exception:
                # Embedding endpoint failed for this query — fall back to keywords.
                pass
        return self._keyword_search(query, top_k)

    def _embedding_search(self, query: str, top_k: int) -> List[Tuple[str, str, float]]:
        resp = self.client.embeddings.create(model=self.embedding_model, input=[query])
        q_vec = resp.data[0].embedding
        scored = []
        for chunk in self.chunks:
            if chunk.embedding is None:
                continue
            score = _cosine(q_vec, chunk.embedding)
            if score > 0.15:  # loose floor; keyword fallback rescues the rest
                scored.append((chunk.source, chunk.text, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[str, str, float]]:
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        idf = self._idf(q_terms)
        scored = []
        for chunk in self.chunks:
            term_counts = {}
            for t in chunk.terms:
                term_counts[t] = term_counts.get(t, 0) + 1
            score = 0.0
            for term in set(q_terms):
                if term in term_counts:
                    score += (1 + math.log(term_counts[term])) * idf.get(term, 1.0)
            if score > 0:
                scored.append((chunk.source, chunk.text, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def _idf(self, terms: List[str]) -> dict:
        total = max(1, len(self.chunks))
        idf = {}
        for term in set(terms):
            containing = sum(1 for c in self.chunks if term in c.terms)
            idf[term] = math.log((total + 1) / (containing + 1)) + 1.0
        return idf


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
