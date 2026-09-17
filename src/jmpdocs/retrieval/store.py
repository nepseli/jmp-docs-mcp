"""Loads the built index and owns the models.

Everything here is lazy and cached: the embedder (~440 MB) and the cross-encoder
reranker (~1.1 GB) are only pulled into memory the first time they are actually
needed, which keeps MCP-server startup fast when a request only needs metadata.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from jmpdocs.config import Settings, get_settings
from jmpdocs.ingest.build_index import (
    BM25_FILE,
    FAISS_FILE,
    Chunk,
    load_chunks,
    load_manifest,
    tokenize,
)

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder, SentenceTransformer

# BGE models want an instruction prefix on the *query* side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class IndexNotBuilt(RuntimeError):
    """Raised when the index files are missing."""


class DocStore:
    """The built corpus: chunks, dense index, lexical index, models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.manifest = load_manifest(self.settings)
        self.chunks: list[Chunk] = load_chunks(self.settings)

        if not self.chunks:
            raise IndexNotBuilt(
                f"no chunks found in {self.settings.index_dir}. "
                "Run: python scripts/build_index.py"
            )

        self._faiss = None
        self._bm25 = None
        self._embedder: "SentenceTransformer | None" = None
        self._reranker: "CrossEncoder | None" = None

        self._by_path: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self._by_path.setdefault(c.path, []).append(i)

    # ---- lazily loaded pieces ------------------------------------------

    @property
    def faiss_index(self):
        if self._faiss is None:
            import faiss

            path = self.settings.index_dir / FAISS_FILE
            if not path.exists():
                raise IndexNotBuilt(f"missing {path}")
            self._faiss = faiss.read_index(str(path))
        return self._faiss

    @property
    def bm25(self):
        if self._bm25 is None:
            path = self.settings.index_dir / BM25_FILE
            if not path.exists():
                raise IndexNotBuilt(f"missing {path}")
            with path.open("rb") as fh:
                self._bm25 = pickle.load(fh)
        return self._bm25

    @property
    def embedder(self) -> "SentenceTransformer":
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(
                self.settings.index.embed_model, device="cpu"
            )
        return self._embedder

    @property
    def reranker(self) -> "CrossEncoder":
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(
                self.settings.index.rerank_model, device="cpu"
            )
        return self._reranker

    # ---- search primitives ---------------------------------------------

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.embedder.encode(
            [BGE_QUERY_PREFIX + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec.astype("float32")

    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        k = min(k, len(self.chunks))
        scores, ids = self.faiss_index.search(self.embed_query(query), k)
        return [
            (int(i), float(s))
            for i, s in zip(ids[0], scores[0])
            if i != -1
        ]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        k = min(k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def rerank(self, query: str, indices: Sequence[int]) -> list[tuple[int, float]]:
        if not indices:
            return []
        # The cross-encoder is 278M params on CPU, so both the number of pairs
        # and the passage length are capped -- they are the dominant cost in a
        # query, not the vector search.
        limit = self.settings.retrieval.rerank_max_chars
        pairs = [
            (
                query,
                f"{self.chunks[i].context_line}\n{self.chunks[i].text}"[:limit],
            )
            for i in indices
        ]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(indices, scores), key=lambda t: t[1], reverse=True)
        return [(int(i), float(s)) for i, s in ranked]

    # ---- lookups --------------------------------------------------------

    def chunks_for_path(self, path: str) -> list[Chunk]:
        return [self.chunks[i] for i in self._by_path.get(path, ())]

    @property
    def paths(self) -> list[str]:
        return list(self._by_path)

    @property
    def books(self) -> list[str]:
        seen: dict[str, int] = {}
        for c in self.chunks:
            if c.book:
                seen[c.book] = seen.get(c.book, 0) + 1
        return sorted(seen, key=lambda b: -seen[b])

    def __len__(self) -> int:
        return len(self.chunks)


@lru_cache(maxsize=1)
def get_store(settings: Settings | None = None) -> DocStore:
    return DocStore(settings)


def index_exists(settings: Settings | None = None) -> bool:
    st = settings or get_settings()
    from jmpdocs.ingest.build_index import CHUNKS_FILE

    return (st.index_dir / CHUNKS_FILE).exists() and (
        st.index_dir / FAISS_FILE
    ).exists()
