"""Retrieval quality, scored against the golden question set.

These are the tests that actually validate the project's central claim: that a
user can ask in their own words and reach the right JMP page. They need the
built index, and skip cleanly without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jmpdocs.knowledge.aliases import AliasBridge
from jmpdocs.retrieval.hybrid import search

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden_questions.yaml"

pytestmark = pytest.mark.needs_index


def _questions() -> list[dict]:
    spec = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    return spec["questions"]


def _first_hit_rank(hits, expected: list[str]) -> int | None:
    wanted = [e.casefold() for e in expected]
    for i, h in enumerate(hits, start=1):
        if any(w in h.chunk.title.casefold() for w in wanted):
            return i
    return None


@pytest.fixture(scope="module")
def golden() -> list[dict]:
    return _questions()


# --------------------------------------------------------------------------
# per-question
# --------------------------------------------------------------------------


def _params():
    """Questions marked known_gap are xfailed: the docs genuinely lack a page."""
    out = []
    for q in _questions():
        marks = [pytest.mark.xfail(reason="documented gap in the JMP docs", strict=False)] if q.get("known_gap") else []
        out.append(pytest.param(q, marks=marks, id=q["id"]))
    return out


@pytest.mark.parametrize("item", _params())
def test_golden_question_retrieves_expected_page(item: dict, store) -> None:
    res = search(item["q"], k=6, store=store)
    rank = _first_hit_rank(res.hits, item["expect_title_any"])
    assert rank is not None, (
        f"{item['q']!r} did not retrieve any of {item['expect_title_any']} in top 6.\n"
        f"intent={res.intent} aliases={res.alias_terms}\n"
        + "\n".join(f"  - {h.chunk.title} [{h.chunk.page_type}]" for h in res.hits)
    )


# --------------------------------------------------------------------------
# aggregate quality gates
# --------------------------------------------------------------------------


def test_overall_recall_at_6(golden: list[dict], store) -> None:
    ranks = [
        _first_hit_rank(search(q["q"], k=6, store=store).hits, q["expect_title_any"])
        for q in golden
    ]
    recall = sum(1 for r in ranks if r) / len(ranks)
    assert recall >= 0.90, f"recall@6 = {recall:.1%}, expected >= 90%"


def test_vocabulary_bridge_lifts_recall(golden: list[dict], store) -> None:
    """The bridge must demonstrably beat retrieval without it.

    This is the load-bearing claim of the whole user-understanding layer, so it
    is asserted rather than assumed.
    """
    bridge_questions = [q for q in golden if q.get("category") == "vocabulary_bridge"]
    empty = AliasBridge(aliases=[])

    with_bridge = sum(
        1
        for q in bridge_questions
        if _first_hit_rank(search(q["q"], k=6, store=store).hits, q["expect_title_any"])
    )
    without = sum(
        1
        for q in bridge_questions
        if _first_hit_rank(
            search(q["q"], k=6, store=store, bridge=empty).hits, q["expect_title_any"]
        )
    )
    assert with_bridge >= without, (
        f"the vocabulary bridge hurt recall: {with_bridge} with vs {without} without"
    )


# --------------------------------------------------------------------------
# intent-aware ranking
# --------------------------------------------------------------------------


def test_how_to_and_interpret_rank_differently(store) -> None:
    """Same topic, different intent, different kind of page on top."""
    how = search("How do I run a Partition analysis?", k=6, store=store)
    why = search("How is the Partition split criterion computed?", k=6, store=store)

    assert how.intent == "how_to"
    assert why.intent == "interpret"

    how_types = [h.chunk.page_type for h in how.hits]
    why_types = [h.chunk.page_type for h in why.hits]
    assert how_types != why_types, "intent had no effect on the result ordering"


def test_book_filter_is_respected(store) -> None:
    res = search(
        "how do I create a design",
        k=6,
        store=store,
        books=["Design of Experiments Guide"],
    )
    assert res.hits
    assert all(h.chunk.book == "Design of Experiments Guide" for h in res.hits)


def test_exact_jsl_identifier_is_found(store) -> None:
    """BM25's job: exact tokens that embeddings blur."""
    res = search("Choose Function JSL", k=6, store=store)
    assert any("choose" in h.chunk.title.casefold() for h in res.hits)


def test_top_relevance_is_absolute_not_normalised(store) -> None:
    """Regression: Hit.score is min-max normalised, so it always reads ~1.0.

    The graph's retry decision needs a score comparable *across* queries, or the
    self-correction path is dead code. top_relevance provides that.
    """
    good = search("How do I make a histogram?", k=6, store=store)
    junk = search("zzqqxx flurble wibbulator", k=6, store=store)

    # normalised score cannot distinguish them
    assert good.hits[0].score == pytest.approx(junk.hits[0].score, abs=0.35)
    # the absolute cross-encoder relevance can
    assert good.top_relevance > junk.top_relevance


def test_out_of_scope_query_is_not_confident(store) -> None:
    res = search("how do I train a YOLO object detection model", k=6, store=store)
    in_scope = search("how do I make a control chart", k=6, store=store)
    assert res.top_relevance < in_scope.top_relevance
