"""Parse the JMP master table of contents.

The whole JMP help hierarchy lives in a single ~2.1 MB file at /19.1/jmp.html,
which means page discovery costs exactly one HTTP request. That file holds two
blocks we care about:

  <div id="toc:GUID">    nested <ul>/<li> tree, ~5,200 WebWorks_TOC_Link anchors.
                         Gives every page, its title, and its position/depth.

  <div id="data:GUID">   contains a <!-- Topics --> run of ~2,900 anchors with
                         ids of the form  topic:<guid>:<UI_WIDGET_NAME>.
                         This is JMP's *context-sensitive help map*: it links the
                         name of a thing on screen ("Query Builder Join Editor")
                         to the exact documentation section that explains it.
                         It is the single best user-vocabulary asset in the site.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

from jmpdocs.config import Settings, get_settings

TOC_CACHE_NAME = "_master_toc.html"


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass(slots=True)
class TocEntry:
    """One clickable line in the help system's table of contents."""

    path: str  # corpus-relative, e.g. "jmp/tabulate.shtml"
    anchor: str  # in-page anchor, e.g. "ww235838" ("" when none)
    title: str
    depth: int  # <ul> nesting depth, 1 = top level
    order: int  # position in a depth-first walk of the TOC
    is_folder: bool  # has children


@dataclass(slots=True)
class UiMapEntry:
    """A JMP on-screen UI element mapped to the docs section describing it."""

    ui_name: str  # "Query Builder Join Editor"
    key: str  # "Query_Builder_Join_Editor"
    path: str
    anchor: str


@dataclass(slots=True)
class PageMeta:
    """Everything the TOC alone can tell us about one documentation page."""

    path: str
    title: str
    depth: int
    order: int
    page_type: str
    is_folder: bool
    toc_titles: list[str] = field(default_factory=list)  # all TOC lines hitting this page
    ui_names: list[str] = field(default_factory=list)  # on-screen names pointing here


# --------------------------------------------------------------------------
# page-type taxonomy
# --------------------------------------------------------------------------

# The docs follow a consistent naming convention that maps almost 1:1 onto user
# intent. Measured across the TOC: 534 "Example(s) of...", 376 "...Options...",
# 264 "Statistical Details...", 119 "Launch the...", 114 "Overview of...".
# Tagging pages with this lets retrieval prefer "Launch the Partition Platform"
# over "Statistical Details for Partition" when the user asks *how do I*.
_PAGE_TYPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("example", re.compile(r"^(additional\s+)?examples?\b|\bexample\s+of\b", re.I)),
    ("launch", re.compile(r"^launch\b", re.I)),
    ("overview", re.compile(r"^(overview|introduction)\b", re.I)),
    ("statistical_details", re.compile(r"^statistical\s+details\b", re.I)),
    (
        "reference",
        re.compile(r"\b(functions?|operators?|messages|syntax reference)\b", re.I),
    ),
    ("options", re.compile(r"\boptions\b", re.I)),
    ("report", re.compile(r"\breports?\b", re.I)),
    (
        "howto",
        re.compile(
            r"^(create|add|import|export|save|use|set|change|edit|remove|delete|"
            r"select|customize|configure|open|run|work with|manage|convert|copy|"
            r"move|sort|filter|join|build|enter|view|show|hide|find|replace)\b",
            re.I,
        ),
    ),
)


def classify_page_type(title: str) -> str:
    """Bucket a page title into the intent taxonomy above."""
    t = (title or "").strip()
    for name, pattern in _PAGE_TYPE_RULES:
        if pattern.search(t):
            return name
    return "concept"


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def fetch_toc_html(settings: Settings | None = None, *, force: bool = False) -> str:
    """Fetch the master TOC, caching it under data/raw/."""
    st = settings or get_settings()
    st.ensure_dirs()
    cache = st.raw_dir / TOC_CACHE_NAME

    if cache.exists() and not force:
        return cache.read_text(encoding="utf-8", errors="replace")

    resp = httpx.get(
        st.source.toc_url,
        timeout=st.crawl.timeout,
        headers={"User-Agent": st.crawl.user_agent},
        follow_redirects=True,
    )
    resp.raise_for_status()
    html = resp.text
    cache.write_text(html, encoding="utf-8")
    return html


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _split_href(href: str, book_root: str) -> tuple[str, str] | None:
    """'jmp/tabulate.shtml#ww235838' -> ('jmp/tabulate.shtml', 'ww235838')."""
    href = (href or "").strip()
    if not href or not href.startswith(f"{book_root}/"):
        return None
    path, _, anchor = href.partition("#")
    if not path.endswith(".shtml"):
        return None
    return path, anchor


def parse_toc(html: str, settings: Settings | None = None) -> list[TocEntry]:
    """Walk the nested <ul>/<li> TOC tree, recording depth and document order."""
    st = settings or get_settings()
    soup = BeautifulSoup(html, "lxml")

    root = soup.find("div", id=re.compile(r"^toc:"))
    if root is None:
        raise ValueError("could not locate the <div id='toc:...'> block in the master TOC")

    entries: list[TocEntry] = []
    counter = 0

    def walk(ul: Tag, depth: int) -> None:
        nonlocal counter
        for li in ul.find_all("li", recursive=False):
            link = li.find("a", class_="WebWorks_TOC_Link")
            child_uls = li.find_all("ul", recursive=False)
            if link is not None:
                split = _split_href(link.get("href", ""), st.source.book_root)
                if split is not None:
                    path, anchor = split
                    entries.append(
                        TocEntry(
                            path=path,
                            anchor=anchor,
                            title=link.get_text(strip=True),
                            depth=depth,
                            order=counter,
                            is_folder=bool(child_uls),
                        )
                    )
                    counter += 1
            for child in child_uls:
                walk(child, depth + 1)

    for ul in root.find_all("ul", recursive=False):
        walk(ul, 1)

    return entries


def parse_ui_map(html: str, settings: Settings | None = None) -> list[UiMapEntry]:
    """Extract JMP's context-sensitive help map (on-screen name -> doc section)."""
    st = settings or get_settings()
    soup = BeautifulSoup(html, "lxml")

    out: list[UiMapEntry] = []
    seen: set[tuple[str, str]] = set()

    for a in soup.find_all("a", id=re.compile(r"^topic:")):
        split = _split_href(a.get("href", ""), st.source.book_root)
        if split is None:
            continue
        path, anchor = split
        # id looks like  topic:<page-guid>:<UI_WIDGET_KEY>
        raw_id = a["id"]
        key = raw_id.split(":", 2)[2] if raw_id.count(":") >= 2 else ""
        name = a.get_text(strip=True) or key.replace("_", " ")
        # A handful of entries pack several comma-separated aliases into one anchor.
        for part in (p.strip() for p in name.split(",")):
            if not part:
                continue
            dedupe = (part.casefold(), path)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            out.append(UiMapEntry(ui_name=part, key=key, path=path, anchor=anchor))

    return out


def build_page_index(
    toc: list[TocEntry], ui_map: list[UiMapEntry]
) -> dict[str, PageMeta]:
    """Collapse TOC entries (many per page) into one record per page."""
    pages: dict[str, PageMeta] = {}

    for e in toc:
        meta = pages.get(e.path)
        if meta is None:
            pages[e.path] = PageMeta(
                path=e.path,
                title=e.title,
                depth=e.depth,
                order=e.order,
                page_type=classify_page_type(e.title),
                is_folder=e.is_folder,
                toc_titles=[e.title],
            )
            continue

        if e.title not in meta.toc_titles:
            meta.toc_titles.append(e.title)
        # Keep the shallowest, earliest appearance as the canonical identity.
        if e.depth < meta.depth or (e.depth == meta.depth and e.order < meta.order):
            meta.title = e.title
            meta.depth = e.depth
            meta.order = e.order
            meta.page_type = classify_page_type(e.title)
        meta.is_folder = meta.is_folder or e.is_folder

    for u in ui_map:
        meta = pages.get(u.path)
        if meta is None:
            # A few UI targets are not reachable from the visible TOC; keep them.
            pages[u.path] = PageMeta(
                path=u.path,
                title=u.ui_name,
                depth=99,
                order=10**9,
                page_type="concept",
                is_folder=False,
                toc_titles=[],
                ui_names=[u.ui_name],
            )
        elif u.ui_name not in meta.ui_names:
            meta.ui_names.append(u.ui_name)

    return pages


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def save_knowledge(
    pages: dict[str, PageMeta],
    ui_map: list[UiMapEntry],
    settings: Settings | None = None,
) -> tuple[Path, Path]:
    """Write pages.json and ui_map.json into data/knowledge/."""
    st = settings or get_settings()
    st.ensure_dirs()

    pages_path = st.knowledge_dir / "pages.json"
    ui_path = st.knowledge_dir / "ui_map.json"

    ordered = sorted(pages.values(), key=lambda m: m.order)
    pages_path.write_text(
        json.dumps([asdict(m) for m in ordered], indent=2), encoding="utf-8"
    )
    ui_path.write_text(
        json.dumps([asdict(u) for u in ui_map], indent=2), encoding="utf-8"
    )
    return pages_path, ui_path


def load_master_toc(
    settings: Settings | None = None, *, force: bool = False
) -> tuple[dict[str, PageMeta], list[UiMapEntry], list[TocEntry]]:
    """One-call entry point: fetch, parse, and index the master TOC."""
    st = settings or get_settings()
    html = fetch_toc_html(st, force=force)
    toc = parse_toc(html, st)
    ui_map = parse_ui_map(html, st)
    return build_page_index(toc, ui_map), ui_map, toc


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    from collections import Counter

    settings = get_settings()
    pages, ui_map, toc = load_master_toc(settings)

    print(f"TOC entries      : {len(toc):,}")
    print(f"UI map entries   : {len(ui_map):,}")
    print(f"unique pages     : {len(pages):,}")
    print(f"max TOC depth    : {max(e.depth for e in toc)}")
    print("\npage_type distribution:")
    for name, count in Counter(m.page_type for m in pages.values()).most_common():
        print(f"  {count:5d}  {name}")

    print("\ntop-level books:")
    for e in toc:
        if e.depth == 1:
            print(f"  {e.title}")

    p, u = save_knowledge(pages, ui_map, settings)
    print(f"\nwrote {p}")
    print(f"wrote {u}")
