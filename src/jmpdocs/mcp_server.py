"""FastMCP server exposing the JMP documentation.

    python -m jmpdocs.mcp_server            # stdio (Claude Desktop / Claude Code)
    python -m jmpdocs.mcp_server --http     # streamable-http, for later deployment

The tools are shaped around how JMP users actually look things up, not around
the shape of the index: `whats_this` answers "what is this thing on my screen"
from JMP's own context-help map, and `find_menu_path` answers "where do I click"
from JMP's own menu reference.
"""

from __future__ import annotations

import argparse
import re
import json
from dataclasses import asdict
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from jmpdocs.config import get_settings
from jmpdocs.ingest.build_index import load_manifest
from jmpdocs.ingest.crawl import fetch_live
from jmpdocs.ingest.menus import load_menus
from jmpdocs.ingest.parse import parse_page
from jmpdocs.knowledge.aliases import get_alias_bridge
from jmpdocs.knowledge.intent import INTENTS
from jmpdocs.retrieval.hybrid import search
from jmpdocs.retrieval.store import get_store, index_exists

mcp = FastMCP(
    name="jmp-docs",
    instructions=(
        "Searches the JMP 19.1 statistical software documentation. Use "
        "search_jmp_docs for general questions, whats_this when the user names "
        "something they see on screen in JMP, and find_menu_path when they ask "
        "where a feature lives. JMP often has its own name for a standard "
        "technique (a random forest is a 'Bootstrap Forest'); the search "
        "handles that translation automatically."
    ),
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _require_index() -> None:
    if not index_exists():
        raise RuntimeError(
            "The JMP documentation index has not been built. Run "
            "`python scripts/build_corpus.py` then `python scripts/build_index.py`."
        )


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _covers(term: str, words: set[str]) -> bool:
    """Whole-word match, tolerating a simple plural/verb suffix."""
    if term in words:
        return True
    return any(w.startswith(term) and len(w) - len(term) <= 2 for w in words)


def _hit_payload(h: Any) -> dict:
    c = h.chunk
    return {
        "title": c.title,
        "book": c.book,
        "breadcrumb": " > ".join(c.breadcrumb),
        "section": c.section,
        "page_type": c.page_type,
        "url": c.url,
        "text": c.text,
        "menu_paths": c.menu_paths,
        "sample_data": c.sample_data,
        "figures": [asdict(f) for f in c.figures],
        "score": round(h.score, 4),
    }


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


@mcp.tool
def search_jmp_docs(
    query: Annotated[str, Field(description="Natural-language question about JMP.")],
    book: Annotated[
        str | None,
        Field(description="Restrict to one book, e.g. 'Design of Experiments Guide'."),
    ] = None,
    intent: Annotated[
        str | None,
        Field(description=f"Optional intent hint. One of: {', '.join(INTENTS)}."),
    ] = None,
    k: Annotated[int, Field(description="Number of excerpts to return.", ge=1, le=20)] = 6,
) -> dict:
    """Search the JMP 19.1 documentation and return the most relevant excerpts.

    Handles JMP's own naming: asking for a "random forest" finds "Bootstrap
    Forest", "pivot table" finds "Tabulate", and so on.
    """
    _require_index()
    result = search(query, intent=intent, books=[book] if book else None, k=k)
    return {
        "query": query,
        "intent": result.intent,
        "jmp_terms_matched": result.alias_terms,
        "n_results": len(result.hits),
        "results": [_hit_payload(h) for h in result.hits],
    }


@mcp.tool
def whats_this(
    ui_element: Annotated[
        str,
        Field(description="Name of something visible in JMP, e.g. 'Query Builder Join Editor'."),
    ],
    k: Annotated[int, Field(description="Max matches.", ge=1, le=10)] = 3,
) -> dict:
    """Explain an on-screen JMP element using JMP's own context-help map.

    JMP ships a mapping from UI element names to the exact documentation
    section that describes them; this looks the user's words up in it.
    """
    _require_index()
    st = get_settings()
    ui_map = json.loads((st.knowledge_dir / "ui_map.json").read_text(encoding="utf-8"))

    needle = ui_element.casefold().strip()
    exact = [u for u in ui_map if u["ui_name"].casefold() == needle]
    partial = [
        u for u in ui_map
        if u not in exact
        and (needle in u["ui_name"].casefold() or u["ui_name"].casefold() in needle)
    ]
    matches = (exact + partial)[:k]

    store = get_store()
    out = []
    for m in matches:
        chunks = store.chunks_for_path(m["path"])
        out.append(
            {
                "ui_name": m["ui_name"],
                "url": st.source.page_url(m["path"])
                + (f"#{m['anchor']}" if m["anchor"] else ""),
                "title": chunks[0].title if chunks else "",
                "book": chunks[0].book if chunks else "",
                "text": chunks[0].text if chunks else "",
            }
        )

    if not out:
        fallback = search(ui_element, intent="where_is_it", k=k)
        return {
            "ui_element": ui_element,
            "exact_ui_match": False,
            "matches": [_hit_payload(h) for h in fallback.hits],
        }
    return {"ui_element": ui_element, "exact_ui_match": True, "matches": out}


@mcp.tool
def find_menu_path(
    task: Annotated[str, Field(description="What the user wants to do, e.g. 'fit a model'.")],
    k: Annotated[int, Field(description="Max menu items.", ge=1, le=15)] = 5,
) -> dict:
    """Find where a feature lives in JMP's menus.

    Backed by JMP's own menu reference (every menu item and its description),
    so the paths are authoritative rather than scraped from prose.
    """
    menus = load_menus()
    if not menus:
        raise RuntimeError("menus.json is missing; run scripts/build_corpus.py")

    bridge = get_alias_bridge()
    alias_terms = bridge.jmp_terms_for(task)

    stop = {
        "the", "a", "an", "and", "or", "for", "with", "how", "can", "into",
        "from", "that", "this", "make", "get", "want", "need", "some", "any",
    }
    content = {w for w in re.findall(r"[a-z0-9]+", task.casefold()) if len(w) > 2} - stop

    scored: list[tuple[float, Any]] = []
    for m in menus:
        if m.platform != "windows":
            continue
        item_l = m.item.casefold()
        desc_l = m.description.casefold()

        # Coverage over whole words, not substrings. Substring matching scores
        # "file" against "pro-file-r", which is how "import an excel file" used
        # to rank Graph > Excel Profiler above File > Import Multiple Files.
        item_words = _words(item_l)
        desc_words = _words(desc_l)
        if content:
            item_cov = sum(1 for w in content if _covers(w, item_words)) / len(content)
            desc_cov = sum(1 for w in content if _covers(w, desc_words)) / len(content)
        else:
            item_cov = desc_cov = 0.0
        score = 6 * item_cov + 3 * desc_cov

        for n in (t.casefold() for t in alias_terms):
            if n == item_l:
                score += 8
            elif n in item_l:
                score += 4
            elif n in desc_l:
                score += 2

        if task.casefold().strip() == item_l:
            score += 10

        if score > 0:
            scored.append((score, m))

    scored.sort(key=lambda t: t[0], reverse=True)
    return {
        "task": task,
        "matches": [
            {
                "menu_path": m.menu_path,
                "menu": m.menu,
                "item": m.item,
                "description": m.description,
            }
            for _, m in scored[:k]
        ],
    }


@mcp.tool
def get_jmp_page(
    page: Annotated[
        str,
        Field(description="Page path ('jmp/tabulate.shtml') or an exact page title."),
    ],
) -> dict:
    """Return the full text of one documentation page, with its figures."""
    _require_index()
    store = get_store()

    path = page if page in store.paths else ""
    if not path:
        needle = page.casefold()
        titles = {c.path: c.title for c in store.chunks}
        for p, t in titles.items():
            if t.casefold() == needle:
                path = p
                break
        if not path:
            for p, t in titles.items():
                if needle in t.casefold():
                    path = p
                    break
    if not path:
        return {"error": f"no page matching {page!r}", "page": page}

    chunks = store.chunks_for_path(path)
    figures = {f.src: asdict(f) for c in chunks for f in c.figures}
    first = chunks[0]
    return {
        "path": path,
        "url": first.url,
        "title": first.title,
        "book": first.book,
        "breadcrumb": " > ".join(first.breadcrumb),
        "page_type": first.page_type,
        "menu_paths": sorted({m for c in chunks for m in c.menu_paths}),
        "sample_data": sorted({s for c in chunks for s in c.sample_data}),
        "figures": list(figures.values()),
        "text": "\n\n".join(c.text for c in chunks),
    }


@mcp.tool
def find_example(
    topic: Annotated[str, Field(description="Topic or JMP platform to find examples for.")],
    dataset: Annotated[
        str | None,
        Field(description="Optional sample data table, e.g. 'Big Class.jmp'."),
    ] = None,
    k: Annotated[int, Field(description="Max examples.", ge=1, le=15)] = 5,
) -> dict:
    """Find worked examples, optionally restricted to a sample data table.

    JMP users learn from the built-in sample data, so this searches the ~630
    'Example of...' pages specifically.
    """
    _require_index()
    result = search(topic, intent="example", page_types=["example"], k=k * 3)
    hits = result.hits
    if dataset:
        needle = dataset.casefold()
        hits = [h for h in hits if any(needle in s.casefold() for s in h.chunk.sample_data)]
    return {
        "topic": topic,
        "dataset": dataset,
        "examples": [_hit_payload(h) for h in hits[:k]],
    }


@mcp.tool
def find_jmp_figures(
    query: Annotated[str, Field(description="What the figure should show.")],
    k: Annotated[int, Field(description="Max figures.", ge=1, le=20)] = 6,
) -> dict:
    """Find documentation figures (JMP dialog and report screenshots) by caption."""
    _require_index()
    st = get_settings()
    result = search(query, k=k * 4)
    seen: set[str] = set()
    figures = []
    for h in result.hits:
        for f in h.chunk.figures:
            if f.src in seen:
                continue
            seen.add(f.src)
            figures.append(
                {
                    "figure_id": f.figure_id,
                    "caption": f.caption,
                    "local_path": str(st.images_dir / f.src.split("/")[-1]),
                    "source_url": st.source.page_url(f.src),
                    "page_title": h.chunk.title,
                    "page_url": h.chunk.url,
                }
            )
            if len(figures) >= k:
                break
        if len(figures) >= k:
            break
    return {"query": query, "figures": figures}


@mcp.tool
def list_jmp_books() -> dict:
    """List the JMP documentation books, with page counts."""
    _require_index()
    store = get_store()
    counts: dict[str, set[str]] = {}
    for c in store.chunks:
        if c.book:
            counts.setdefault(c.book, set()).add(c.path)
    return {
        "books": [
            {"book": b, "pages": len(p)}
            for b, p in sorted(counts.items(), key=lambda t: -len(t[1]))
        ]
    }


@mcp.tool
def browse_jmp_toc(
    book: Annotated[str, Field(description="Book name from list_jmp_books.")],
    contains: Annotated[
        str | None, Field(description="Only titles containing this text.")
    ] = None,
    limit: Annotated[int, Field(description="Max entries.", ge=1, le=200)] = 60,
) -> dict:
    """Browse the table of contents for one book."""
    _require_index()
    store = get_store()
    seen: set[str] = set()
    entries = []
    for c in store.chunks:
        if c.book != book or c.path in seen:
            continue
        if contains and contains.casefold() not in c.title.casefold():
            continue
        seen.add(c.path)
        entries.append(
            {
                "title": c.title,
                "path": c.path,
                "page_type": c.page_type,
                "breadcrumb": " > ".join(c.breadcrumb),
            }
        )
        if len(entries) >= limit:
            break
    return {"book": book, "n_entries": len(entries), "entries": entries}


@mcp.tool
def jmp_fetch_live(
    url: Annotated[
        str,
        Field(description="A https://www.jmp.com/support/help/en/19.1/... URL."),
    ],
) -> dict:
    """Fetch a documentation page live from jmp.com, bypassing the local index.

    Use when a page may be newer than the index, or is not indexed.
    """
    st = get_settings()
    html = fetch_live(url, st)
    rel = url.replace(st.source.base_url, "").lstrip("/").split("#")[0]
    doc = parse_page(html, rel, None, st)
    return {
        "url": url,
        "title": doc.title,
        "book": doc.book,
        "breadcrumb": " > ".join(doc.breadcrumb),
        "published": doc.published,
        "text": doc.markdown,
        "figures": [asdict(f) for f in doc.figures],
    }


@mcp.tool
def index_stats() -> dict:
    """Report what the local index contains and when it was built."""
    st = get_settings()
    manifest = load_manifest(st)
    if not manifest:
        return {"built": False, "message": "index not built"}
    return {"built": True, "data_dir": str(st.data_dir), **manifest}


# --------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------


@mcp.resource("jmp://page/{path*}")
def page_resource(path: str) -> str:
    """Full Markdown of one documentation page."""
    return get_jmp_page(path).get("text", "")


@mcp.resource("jmp://book/{book}")
def book_resource(book: str) -> str:
    """Table of contents for one book, as Markdown."""
    data = browse_jmp_toc(book, limit=200)
    lines = [f"# {book}", ""]
    lines += [f"- {e['title']}  ({e['page_type']})" for e in data["entries"]]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="JMP documentation MCP server")
    ap.add_argument("--http", action="store_true", help="serve over streamable-http")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.http:
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
