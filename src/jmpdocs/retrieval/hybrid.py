"""Hybrid retrieval: dense + lexical, reranked, with intent-aware boosting.

Pipeline for one question:

    expanded queries ->  dense (FAISS)  +  lexical (BM25)
                                |
                       min-max normalise each, weighted fuse
                                |
                         cross-encoder rerank
                                |
                    soft boost for intent-matching page types
                                |
                              top k

Why both retrievers: BM25 is what actually resolves exact tokens -- `Choose()`,
"Bootstrap Forest" -- while embeddings handle paraphrase. Fusing them is what
makes the vocabulary bridge pay off, since the bridge's synonyms are injected
into the BM25 side of the index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from jmpdocs.config import Settings, get_settings
from jmpdocs.ingest.build_index import Chunk
from jmpdocs.knowledge.aliases import AliasBridge, get_alias_bridge
from jmpdocs.knowledge.intent import (
    DEFAULT_INTENT,
    SCRIPTING_BOOKS,
    classify_intent,
    page_type_weight,
)
from jmpdocs.retrieval.store import DocStore, get_store


@dataclass(slots=True)
class Hit:
    chunk: Chunk
    score: float
    dense_score: float = 0.0  # normalised within this query
    bm25_score: float = 0.0  # normalised within this query
    dense_cosine: float = 0.0  # raw cosine -- comparable *across* queries
    rerank_score: float = 0.0
    boost: float = 0.0

    @property
    def citation(self) -> str:
        trail = " > ".join(self.chunk.breadcrumb) or self.chunk.book
        return trail or self.chunk.title


@dataclass(slots=True)
class SearchResult:
    query: str
    intent: str
    queries_used: list[str]
    alias_terms: list[str]
    hits: list[Hit] = field(default_factory=list)
    reranked: bool = False

    @property
    def top_relevance(self) -> float:
        """An *absolute* confidence signal, unlike `Hit.score`.

        Hit.score is min-max normalised, so the best hit always reads ~1.0 even
        when nothing relevant was found -- useless for deciding whether to retry.
        Both signals below are comparable across queries: the cross-encoder logit
        (roughly >0 means relevant) and, when reranking is off, the raw cosine
        from the normalised-embedding index.
        """
        if not self.hits:
            return float("-inf")
        return (
            self.hits[0].rerank_score if self.reranked else self.hits[0].dense_cosine
        )

    @property
    def relevance_floor(self) -> float:
        """Threshold below which `top_relevance` means 'not covered'."""
        return WEAK_RERANK_LOGIT if self.reranked else WEAK_DENSE_COSINE

    @property
    def figures(self) -> list:
        seen: set[str] = set()
        out = []
        for h in self.hits:
            for f in h.chunk.figures:
                if f.src not in seen:
                    seen.add(f.src)
                    out.append(f)
        return out


# Absolute relevance floors, established from the eval set.
WEAK_RERANK_LOGIT = -4.0
WEAK_DENSE_COSINE = 0.45


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def search(
    query: str,
    *,
    queries: Sequence[str] | None = None,
    intent: str | None = None,
    books: Iterable[str] | None = None,
    page_types: Iterable[str] | None = None,
    k: int | None = None,
    rerank: bool | None = None,
    store: DocStore | None = None,
    settings: Settings | None = None,
    bridge: AliasBridge | None = None,
) -> SearchResult:
    st = settings if settings is not None else get_settings()
    # `is not None`, not `or`: DocStore and AliasBridge both define __len__, so
    # an empty one is falsy and `x or default()` silently substitutes the real
    # object -- which made the --no-alias ablation measure nothing at all.
    ds = store if store is not None else get_store()
    br = bridge if bridge is not None else get_alias_bridge()

    # Reranking is off by default: on the golden set it matched recall exactly
    # while costing ~10x the latency, once per-query normalisation fixed the
    # fusion. Kept available for harder corpora.
    if rerank is None:
        rerank = st.retrieval.rerank
    resolved_intent = intent or classify_intent(query)
    alias_terms = br.jmp_terms_for(query)

    # The bridge's JMP terms ride along as an extra query so the lexical side
    # gets a shot at the page's real vocabulary ("Bootstrap Forest").
    all_queries = list(queries) if queries else [query]
    if alias_terms:
        all_queries.append(" ".join(alias_terms))
    all_queries = list(dict.fromkeys(q for q in all_queries if q and q.strip()))

    k_final = k or st.retrieval.k_final
    k_each = st.retrieval.k_per_query

    # Normalise each query's results *before* combining them.
    #
    # BM25 scores are not comparable across queries: a long question scores
    # 25-plus simply by having more terms, while the short alias query "Stack"
    # tops out around 10 even when it ranks the correct page first. Pooling raw
    # scores and normalising once lets the long query swamp the alias query
    # completely -- which silently defeated the whole vocabulary bridge.
    dense_n: dict[int, float] = {}
    bm25_n: dict[int, float] = {}
    dense_cos: dict[int, float] = {}
    for q in all_queries:
        raw_dense = dict(ds.dense_search(q, k_each))
        for idx, sc in raw_dense.items():
            dense_cos[idx] = max(dense_cos.get(idx, 0.0), sc)
        for idx, sc in _minmax(raw_dense).items():
            dense_n[idx] = max(dense_n.get(idx, 0.0), sc)
        for idx, sc in _minmax(dict(ds.bm25_search(q, k_each))).items():
            bm25_n[idx] = max(bm25_n.get(idx, 0.0), sc)

    fused: dict[int, float] = {}
    for idx in set(dense_n) | set(bm25_n):
        fused[idx] = (
            st.retrieval.weight_dense * dense_n.get(idx, 0.0)
            + st.retrieval.weight_bm25 * bm25_n.get(idx, 0.0)
        )

    # metadata filters
    book_set = {b for b in (books or ()) if b}
    type_set = {t for t in (page_types or ()) if t}
    if book_set:
        fused = {i: s for i, s in fused.items() if ds.chunks[i].book in book_set}
    if type_set:
        fused = {i: s for i, s in fused.items() if ds.chunks[i].page_type in type_set}

    if not fused:
        return SearchResult(query, resolved_intent, all_queries, alias_terms, [], rerank)

    candidates = sorted(fused, key=lambda i: fused[i], reverse=True)[
        : st.retrieval.rerank_candidates
    ]

    rerank_scores: dict[int, float] = {}
    if rerank:
        # The cross-encoder does not know JMP's vocabulary either, so it will
        # happily demote the right page for asking about "unpivot" when the page
        # only ever says "Stack". Carry the bridge into the rerank query too.
        rerank_query = f"{query} ({', '.join(alias_terms)})" if alias_terms else query
        for idx, sc in ds.rerank(rerank_query, candidates):
            rerank_scores[idx] = sc
        base = _minmax(rerank_scores)
    else:
        base = {i: fused[i] for i in candidates}

    # Soft, intent-aware nudge toward the right *kind* of page. Deliberately not
    # a filter: intent classification is a guess, and a wrong guess should cost
    # a little ranking rather than hide the correct answer.
    boost_weight = st.retrieval.page_type_boost
    final: dict[int, float] = {}
    boosts: dict[int, float] = {}
    for idx, sc in base.items():
        chunk = ds.chunks[idx]
        b = page_type_weight(resolved_intent, chunk.page_type)
        if resolved_intent == "scripting" and chunk.book in SCRIPTING_BOOKS:
            b = max(b, 1.0)
        boosts[idx] = b
        final[idx] = sc + boost_weight * b

    ordered = sorted(final, key=lambda i: final[i], reverse=True)[:k_final]

    hits = [
        Hit(
            chunk=ds.chunks[i],
            score=final[i],
            dense_score=dense_n.get(i, 0.0),
            bm25_score=bm25_n.get(i, 0.0),
            dense_cosine=dense_cos.get(i, 0.0),
            rerank_score=rerank_scores.get(i, 0.0),
            boost=boosts.get(i, 0.0),
        )
        for i in ordered
    ]
    return SearchResult(query, resolved_intent, all_queries, alias_terms, hits, rerank)
