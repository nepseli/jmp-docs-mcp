"""MCP tool tests. Tools needing the index are marked and skip without it."""

from __future__ import annotations

import pytest

from jmpdocs.mcp_server import (
    _covers,
    _words,
    browse_jmp_toc,
    find_menu_path,
    get_jmp_page,
    index_stats,
    list_jmp_books,
    search_jmp_docs,
    whats_this,
)


# --------------------------------------------------------------------------
# menu lookup (needs only menus.json)
# --------------------------------------------------------------------------


def test_word_coverage_is_boundary_aware() -> None:
    """Regression: 'file' must not match inside 'profiler'."""
    assert not _covers("file", _words("Excel Profiler"))
    assert _covers("file", _words("Import Multiple Files"))
    assert _covers("import", _words("Import as Data"))
    assert _covers("design", _words("Custom Design"))


@pytest.mark.parametrize(
    "task, expected",
    [
        ("fit a model", "Analyze > Fit Model"),
        ("make a pivot table", "Analyze > Tabulate"),
        ("design an experiment", "DOE > Custom Design"),
        ("recode messy categories", "Cols > Recode..."),
        ("connect to a database", "File > Connect To"),
        # regression: substring matching used to put Graph > Excel Profiler first
        ("import an excel file", "File > Import Multiple Files..."),
    ],
)
def test_find_menu_path(task: str, expected: str) -> None:
    paths = [m["menu_path"] for m in find_menu_path(task, k=3)["matches"]]
    assert paths, f"no menu match for {task!r}"
    assert paths[0] == expected, f"{task!r} -> {paths}"


def test_find_menu_path_uses_the_vocabulary_bridge() -> None:
    """'random forest' is not a JMP menu label; the bridge has to carry it."""
    paths = [m["menu_path"] for m in find_menu_path("random forest", k=3)["matches"]]
    assert any("Predictive Modeling" in p for p in paths), paths


# --------------------------------------------------------------------------
# index-backed tools
# --------------------------------------------------------------------------


@pytest.mark.needs_index
def test_index_stats_reports_a_built_index(store) -> None:
    stats = index_stats()
    assert stats["built"] is True
    assert stats["n_chunks"] > 1000
    assert stats["jmp_version"] == "19.1"


@pytest.mark.needs_index
def test_list_books(store) -> None:
    books = {b["book"] for b in list_jmp_books()["books"]}
    assert "Scripting Guide" in books
    assert "Design of Experiments Guide" in books


@pytest.mark.needs_index
def test_search_returns_citations_and_alias_matches(store) -> None:
    out = search_jmp_docs("how do I run a random forest", k=5)
    assert out["results"]
    assert "Bootstrap Forest" in out["jmp_terms_matched"]
    for r in out["results"]:
        assert r["url"].startswith("https://www.jmp.com/support/help/en/19.1/")
        assert r["title"]


@pytest.mark.needs_index
def test_get_page_by_path_and_by_title(store) -> None:
    by_path = get_jmp_page("jmp/tabulate.shtml")
    assert by_path["title"] == "Tabulate"
    assert by_path["text"]
    by_title = get_jmp_page("Tabulate")
    assert by_title["path"] == "jmp/tabulate.shtml"


@pytest.mark.needs_index
def test_get_page_reports_unknown_page(store) -> None:
    assert "error" in get_jmp_page("no such page exists here")


@pytest.mark.needs_index
def test_whats_this_hits_the_ui_map(store) -> None:
    out = whats_this("Query Builder Join Editor")
    assert out["exact_ui_match"] is True
    assert out["matches"]
    assert "select-tables-from-a-sql-database" in out["matches"][0]["url"]


@pytest.mark.needs_index
def test_whats_this_falls_back_to_search(store) -> None:
    out = whats_this("some control that does not exist in JMP")
    assert out["exact_ui_match"] is False


@pytest.mark.needs_index
def test_browse_toc_filters_by_book(store) -> None:
    out = browse_jmp_toc("Basic Analysis", limit=25)
    assert out["entries"]
    assert all(e["breadcrumb"].startswith("Basic Analysis") for e in out["entries"])
