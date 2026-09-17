"""User intent taxonomy, and how it maps onto the docs' own page types.

The JMP docs follow a naming convention that lines up almost exactly with what
a user is trying to do. Counted across the real TOC: 634 "Example(s) of...",
411 how-to titles, 294 "...Report", 270 "Statistical Details...", 257
"...Options...", 150 reference pages, 130 "Launch the...", 120 "Overview of...".

Someone asking *"how do I run a Partition?"* wants **Launch the Partition
Platform**; they do not want **Statistical Details for Partition**. Text
similarity alone cannot tell those apart -- both are dense with the word
"Partition". So intent is classified, then used to *boost* matching page types
during reranking.

The boost is deliberately soft (config: retrieval.page_type_boost). A hard
filter would be brittle: intent classification is a guess, and a wrong guess
should cost a little ranking, not hide the right answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# intent -> {page_type: relative weight in 0..1}
INTENT_PAGE_TYPES: dict[str, dict[str, float]] = {
    # "concept" earns a real share here: the canonical page for a feature is
    # usually typed concept ("Life Distribution", "Stack Columns in Data
    # Tables"), and without a weight it gets buried under tangential Example
    # pages that happen to mention the same words.
    "how_to": {"launch": 1.0, "howto": 1.0, "example": 0.6, "concept": 0.45},
    "what_is": {"overview": 1.0, "concept": 0.8},
    "interpret": {"report": 1.0, "statistical_details": 0.8, "options": 0.3},
    "option_lookup": {"options": 1.0, "report": 0.5, "reference": 0.3},
    "example": {"example": 1.0, "howto": 0.4},
    "scripting": {"reference": 1.0, "concept": 0.2},
    "where_is_it": {"launch": 0.9, "howto": 0.7, "overview": 0.3},
    "troubleshoot": {"report": 0.5, "concept": 0.4, "statistical_details": 0.3},
}

DEFAULT_INTENT = "how_to"
INTENTS = tuple(INTENT_PAGE_TYPES)

# Books that are the scripting reference; a scripting intent should favour them.
SCRIPTING_BOOKS = ("Scripting Guide", "JSL Syntax Reference")


@dataclass(slots=True)
class IntentRule:
    intent: str
    pattern: re.Pattern[str]


# Ordered: the first match wins, so put the most specific patterns first.
_RULES: tuple[IntentRule, ...] = (
    IntentRule("scripting", re.compile(
        r"\b(jsl|script|scripting|programmatic\w*|automate|automation|"
        r"function\s+syntax|write\s+code|api|message|send\s*\(|add-?in)\b", re.I)),
    IntentRule("where_is_it", re.compile(
        r"\b(where\s+(is|are|do|can)|which\s+menu|what\s+menu|find\s+the\s+"
        r"(button|option|menu|setting)|how\s+do\s+i\s+get\s+to|located)\b", re.I)),
    IntentRule("option_lookup", re.compile(
        r"\b(what\s+does\s+.*\s+(option|button|check\s*box|setting)|"
        r"red\s+triangle|platform\s+options|what\s+is\s+the\s+.*\s+option|"
        r"options?\s+(menu|available|list))\b", re.I)),
    IntentRule("troubleshoot", re.compile(
        r"\b(why\s+(is|does|do|am|can'?t|won'?t)|error|fails?|failing|not\s+working|"
        r"missing|blank|empty|grey(ed)?\s*out|disabled|unexpected|wrong|problem|"
        r"troubleshoot)\b", re.I)),
    IntentRule("interpret", re.compile(
        r"\b(interpret|meaning|what\s+does\s+.*\s+mean|how\s+do\s+i\s+read|"
        r"understand\s+the\s+(output|report|result)|p-?value|significan\w*|"
        r"how\s+is\s+.*\s+(computed|calculated)|formula\s+for|statistical\s+details)\b", re.I)),
    IntentRule("example", re.compile(
        r"\b(example|walk\s*me\s*through|step\s*by\s*step|tutorial|demo|"
        r"show\s+me\s+how|worked\s+example|sample\s+data)\b", re.I)),
    IntentRule("what_is", re.compile(
        r"^\s*(what\s+is|what\s+are|what'?s|explain|describe|overview\s+of|"
        r"tell\s+me\s+about|when\s+(should|would)\s+i\s+use|difference\s+between)\b", re.I)),
    IntentRule("how_to", re.compile(
        r"\b(how\s+(do|can|would)\s+i|how\s+to|steps?\s+to|create|make|build|"
        r"run|perform|generate|add|set\s+up|configure|import|export|save)\b", re.I)),
)


def classify_intent(question: str) -> str:
    """Best-effort intent for a question, using cheap deterministic rules.

    The graph may override this with an LLM classification, but this rule pass
    is free, instant, and correct often enough to be a sound fallback.
    """
    q = (question or "").strip()
    if not q:
        return DEFAULT_INTENT
    for rule in _RULES:
        if rule.pattern.search(q):
            return rule.intent
    return DEFAULT_INTENT


def page_type_weight(intent: str, page_type: str) -> float:
    """Relative preference in 0..1 for a page type given an intent."""
    return INTENT_PAGE_TYPES.get(intent, {}).get(page_type, 0.0)


def prefers_scripting(intent: str) -> bool:
    return intent == "scripting"
