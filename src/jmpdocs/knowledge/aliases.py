"""The JMP vocabulary bridge.

See aliases.yaml for the data and the reasoning. In short: JMP calls a random
forest a "Bootstrap Forest", and the phrase "random forest" appears nowhere on
that page, so a user's own words retrieve nothing without help.

The bridge is applied at both ends:

    index time   :meth:`AliasBridge.user_terms_for` -- given a chunk that talks
                  about "Bootstrap Forest", return ["random forest", ...] so
                  those words get appended to the chunk's BM25 text.

    query time   :meth:`AliasBridge.jmp_terms_for` -- given the question
                  "how do I run a random forest", return ["Bootstrap Forest"]
                  to inject into the expanded queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ALIAS_FILE = Path(__file__).with_name("aliases.yaml")


@dataclass(slots=True)
class Alias:
    category: str
    jmp: list[str]
    user: list[str]


def _build_matcher(
    phrases_by_index: list[list[str]],
) -> tuple[re.Pattern[str] | None, dict[str, list[int]]]:
    """Compile one alternation over every phrase, plus phrase -> alias indices."""
    lookup: dict[str, list[int]] = {}
    for idx, phrases in enumerate(phrases_by_index):
        for p in phrases:
            key = p.casefold().strip()
            if key:
                lookup.setdefault(key, []).append(idx)

    if not lookup:
        return None, lookup

    # longest first so "random forests" wins over "random forest"
    ordered = sorted(lookup, key=len, reverse=True)
    pattern = "|".join(re.escape(p) for p in ordered)
    return re.compile(rf"(?<!\w)(?:{pattern})(?!\w)", re.I), lookup


@dataclass
class AliasBridge:
    aliases: list[Alias] = field(default_factory=list)

    _jmp_re: re.Pattern[str] | None = field(default=None, repr=False)
    _jmp_lookup: dict[str, list[int]] = field(default_factory=dict, repr=False)
    _user_re: re.Pattern[str] | None = field(default=None, repr=False)
    _user_lookup: dict[str, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._jmp_re, self._jmp_lookup = _build_matcher([a.jmp for a in self.aliases])
        self._user_re, self._user_lookup = _build_matcher([a.user for a in self.aliases])

    # ---- construction ---------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> AliasBridge:
        src = path or ALIAS_FILE
        raw = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
        entries = [
            Alias(
                category=str(e.get("category", "general")),
                jmp=[str(x) for x in e.get("jmp", [])],
                user=[str(x) for x in e.get("user", [])],
            )
            for e in raw.get("aliases", [])
        ]
        return cls(aliases=entries)

    # ---- index time -----------------------------------------------------

    def user_terms_for(self, text: str, limit: int = 24) -> list[str]:
        """User-facing synonyms for whichever JMP terms appear in `text`."""
        if self._jmp_re is None or not text:
            return []
        hits: list[str] = []
        seen: set[str] = set()
        for match in self._jmp_re.finditer(text):
            for idx in self._jmp_lookup.get(match.group(0).casefold(), ()):
                for term in self.aliases[idx].user:
                    key = term.casefold()
                    if key not in seen:
                        seen.add(key)
                        hits.append(term)
                        if len(hits) >= limit:
                            return hits
        return hits

    # ---- query time -----------------------------------------------------

    def jmp_terms_for(self, query: str, limit: int = 8) -> list[str]:
        """JMP terminology implied by the words in a user's question."""
        if self._user_re is None or not query:
            return []
        hits: list[str] = []
        seen: set[str] = set()
        for match in self._user_re.finditer(query):
            for idx in self._user_lookup.get(match.group(0).casefold(), ()):
                for term in self.aliases[idx].jmp:
                    key = term.casefold()
                    if key not in seen:
                        seen.add(key)
                        hits.append(term)
                        if len(hits) >= limit:
                            return hits
        return hits

    def matched_aliases(self, query: str) -> list[Alias]:
        """Alias entries triggered by a query (for explaining the bridge in the UI)."""
        if self._user_re is None or not query:
            return []
        out: list[Alias] = []
        seen: set[int] = set()
        for match in self._user_re.finditer(query):
            for idx in self._user_lookup.get(match.group(0).casefold(), ()):
                if idx not in seen:
                    seen.add(idx)
                    out.append(self.aliases[idx])
        return out

    def __len__(self) -> int:
        return len(self.aliases)


@lru_cache(maxsize=1)
def get_alias_bridge() -> AliasBridge:
    return AliasBridge.load()


if __name__ == "__main__":  # pragma: no cover
    bridge = get_alias_bridge()
    n_user = sum(len(a.user) for a in bridge.aliases)
    n_jmp = sum(len(a.jmp) for a in bridge.aliases)
    print(f"{len(bridge)} alias entries  ({n_user} user terms -> {n_jmp} JMP terms)")

    for q in [
        "how do I run a random forest in JMP",
        "make a pivot table",
        "what is the JMP equivalent of vlookup",
        "check if my process is capable",
        "how do I do a gage r&r",
        "where is the red triangle menu",
    ]:
        print(f"  {q!r}\n     -> {bridge.jmp_terms_for(q)}")

    print("\nindex-time direction:")
    for t in ["The Bootstrap Forest platform fits many trees",
              "Use Tabulate to build summary tables"]:
        print(f"  {t!r}\n     -> {bridge.user_terms_for(t)[:6]}")
