"""Parser tests, run against real cached pages where available."""

from __future__ import annotations

import pytest

from jmpdocs.ingest.crawl import read_cached
from jmpdocs.ingest.parse import (
    _extract_menu_paths,
    _extract_sample_data,
    _normalize_img_src,
    _trim_menu_segment,
    _trim_sample_name,
    parse_page,
)
from jmpdocs.ingest.toc import classify_page_type


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("../jmp/images/1-103.png", "jmp/images/1-103.png"),
        ("../../jmp/images/a.png", "jmp/images/a.png"),
        ("jmp/images/paste%20new.png", "jmp/images/paste new.png"),
    ],
)
def test_normalize_img_src(raw: str, expected: str) -> None:
    assert _normalize_img_src(raw) == expected


@pytest.mark.parametrize(
    "segment, expected",
    [
        ("Distribution. 1. 2. Select profit", "Distribution"),
        ("Sample Data Folder and open the file", "Sample Data Folder"),
        ("Fit Y by X", "Fit Y by X"),
        ("Save Script to Data Table", "Save Script to Data Table"),
    ],
)
def test_trim_menu_segment(segment: str, expected: str) -> None:
    assert _trim_menu_segment(segment) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Sample Data Folder and open Companies.jmp", "Companies.jmp"),
        ("1. 3. In Pizza Responses.jmp", "Pizza Responses.jmp"),
        ("the Drug.jmp", "Drug.jmp"),
        ("Big Class.jmp", "Big Class.jmp"),
    ],
)
def test_trim_sample_name(raw: str, expected: str) -> None:
    assert _trim_sample_name(raw) == expected


def test_extract_menu_paths_stops_at_prose() -> None:
    text = "Select Analyze > Distribution. Then choose a column."
    assert _extract_menu_paths(text) == ["Analyze > Distribution"]


def test_extract_menu_paths_keeps_submenus() -> None:
    text = "Choose Analyze > Quality and Process > Control Chart Builder to begin."
    assert "Analyze > Quality and Process > Control Chart Builder" in _extract_menu_paths(text)


def test_extract_sample_data() -> None:
    text = "Open the Sample Data Folder and open Big Class.jmp for this example."
    assert _extract_sample_data(text) == ["Big Class.jmp"]


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Example of the Tabulate Platform", "example"),
        ("Launch the Fit Model Platform", "launch"),
        ("Overview of the Factor Analysis Platform", "overview"),
        ("Statistical Details for the Logistic Platform", "statistical_details"),
        ("Choose Function", "reference"),
        ("Tabulate Platform Options", "options"),
        ("Create a Subset Data Table", "howto"),
    ],
)
def test_page_type_classification(title: str, expected: str) -> None:
    assert classify_page_type(title) == expected


# --------------------------------------------------------------------------
# real pages
# --------------------------------------------------------------------------


def _cached(path: str, settings):
    html = read_cached(path, settings)
    if html is None:
        pytest.skip(f"{path} not cached; run scripts/build_corpus.py")
    return parse_page(html, path, None, settings)


def test_jsl_reference_page(settings) -> None:
    doc = _cached("jmp/choose-function.shtml", settings)
    assert doc.title == "Choose Function"
    assert doc.book == "Scripting Guide"
    assert doc.page_type == "reference"
    assert doc.published
    # code is fenced, and consecutive one-line <pre> blocks are merged
    assert "```jsl" in doc.markdown
    assert "| x = |" in doc.markdown  # the step table survived as Markdown
    assert "**Note:** Note:" not in doc.markdown  # no doubled Note label


def test_platform_page_figures_and_nav_stripping(settings) -> None:
    doc = _cached("jmp/tabulate.shtml", settings)
    assert doc.title == "Tabulate"
    assert doc.breadcrumb[:1] == ["Basic Analysis"]
    assert doc.figures, "expected at least one figure"
    fig = doc.figures[0]
    assert fig.figure_id.startswith("Figure")
    assert fig.src.startswith("jmp/images/")
    assert "[FIGURE:" in doc.markdown
    # the 19 auto-generated in-page TOC blocks must be gone
    assert "ChapterTOC" not in doc.markdown
    assert "Example of the Tabulate Platform" not in doc.markdown
    assert doc.next  # rel=Next preserved


def test_menu_reference_page_parses(settings) -> None:
    from jmpdocs.ingest.menus import parse_menu_page

    html = read_cached("jmp/jmp-19-windows-menu-descriptions.shtml", settings)
    if html is None:
        pytest.skip("menu page not cached")
    items = parse_menu_page(html, "windows")
    paths = {i.menu_path for i in items}
    # regression guard: the <h2> lives inside each table's <caption>, so a
    # naive document-order walk shifts every menu by one
    assert "Analyze > Distribution" in paths
    assert "Analyze > Fit Y by X" in paths
    assert "DOE > Custom Design" in paths
    assert "Graph > Graph Builder" in paths
    assert not any(p.startswith("Analyze > Graph Builder") for p in paths)
