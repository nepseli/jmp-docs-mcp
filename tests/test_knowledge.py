"""Unit tests for the user-understanding layer: aliases and intent."""

from __future__ import annotations

import pytest

from jmpdocs.knowledge.aliases import AliasBridge, get_alias_bridge
from jmpdocs.knowledge.intent import (
    INTENT_PAGE_TYPES,
    classify_intent,
    page_type_weight,
)


@pytest.fixture(scope="module")
def bridge() -> AliasBridge:
    return get_alias_bridge()


# --------------------------------------------------------------------------
# vocabulary bridge
# --------------------------------------------------------------------------


def test_alias_file_loads(bridge: AliasBridge) -> None:
    assert len(bridge) > 100
    for a in bridge.aliases:
        assert a.jmp, f"{a.category}: entry has no JMP terms"
        assert a.user, f"{a.category}: entry has no user terms"


@pytest.mark.parametrize(
    "question, expected",
    [
        ("How do I run a random forest in JMP?", "Bootstrap Forest"),
        ("Does JMP do gradient boosting?", "Boosted Tree"),
        ("I want a decision tree", "Partition"),
        ("How do I make a pivot table?", "Tabulate"),
        ("What is the JMP equivalent of VLOOKUP?", "Join"),
        ("How do I check if my process is capable?", "Process Capability"),
        ("How do I do a Gage R&R?", "Variability Chart"),
        ("weibull analysis of failure data", "Life Distribution"),
        ("how do I do PCA", "Principal Components"),
        ("k-means clustering", "K Means Cluster"),
        ("where is the red triangle menu", "red triangle menu"),
        ("how do I unpivot from wide to long", "Stack"),
    ],
)
def test_user_words_map_to_jmp_words(
    bridge: AliasBridge, question: str, expected: str
) -> None:
    """The core promise: a user's own vocabulary resolves to JMP's."""
    assert expected in bridge.jmp_terms_for(question), (
        f"{question!r} did not map to {expected!r}; got {bridge.jmp_terms_for(question)}"
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The Bootstrap Forest platform fits many trees", "random forest"),
        ("Use Tabulate to build summary tables", "pivot table"),
        ("The Partition platform splits data", "decision tree"),
    ],
)
def test_jmp_words_map_back_to_user_words(
    bridge: AliasBridge, text: str, expected: str
) -> None:
    """Index-time direction: chunks gain the synonyms BM25 needs."""
    assert expected in bridge.user_terms_for(text)


def test_no_match_returns_empty(bridge: AliasBridge) -> None:
    assert bridge.jmp_terms_for("how do I train a YOLO object detector") == []


def test_matching_is_word_bounded(bridge: AliasBridge) -> None:
    # "cp" is an alias for Process Capability but must not fire inside a word
    assert "Process Capability" not in bridge.jmp_terms_for("copy the column")


# --------------------------------------------------------------------------
# intent
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question, expected",
    [
        ("How do I run a Partition analysis?", "how_to"),
        ("How is the Partition split criterion computed?", "interpret"),
        ("What is the Distribution platform?", "what_is"),
        ("Show me a worked example of Tabulate", "example"),
        ("What does the Choose function do in JSL?", "scripting"),
        ("Where is the red triangle menu?", "where_is_it"),
        ("Why is my p-value blank?", "troubleshoot"),
    ],
)
def test_intent_classification(question: str, expected: str) -> None:
    assert classify_intent(question) == expected


def test_intent_falls_back_for_empty() -> None:
    assert classify_intent("") == "how_to"


def test_how_to_prefers_launch_over_statistical_details() -> None:
    """The routing that makes 'how do I' return steps rather than maths."""
    assert page_type_weight("how_to", "launch") > page_type_weight(
        "how_to", "statistical_details"
    )
    assert page_type_weight("interpret", "statistical_details") > page_type_weight(
        "interpret", "launch"
    )


def test_every_intent_has_page_type_preferences() -> None:
    for intent, mapping in INTENT_PAGE_TYPES.items():
        assert mapping, f"{intent} has no page-type preferences"
        assert all(0 < w <= 1 for w in mapping.values())
