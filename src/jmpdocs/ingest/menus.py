"""Extract JMP's authoritative menu reference.

JMP ships a single page -- "Windows Menu Descriptions" -- that documents every
menu and every item under it, as `Item | Description` tables. That is a much
better source for "where do I find X?" than regex-scraping menu paths out of
prose, and the descriptions are written in task language ("Creates a new data
table...", "Locates and opens files...") which is close to how users phrase
questions.

Structural note: each menu's <h2> heading lives inside that table's own
<caption>, so a flattened document-order walk yields the table *before* its
heading. The menu name therefore comes from the table's caption, not from the
preceding heading -- getting this wrong silently shifts every menu by one.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from jmpdocs.config import Settings, get_settings
from jmpdocs.ingest.crawl import fetch_page

WINDOWS_MENU_PAGE = "jmp/jmp-19-windows-menu-descriptions.shtml"
MACOS_MENU_PAGE = "jmp/jmp-19-apple-macos-menu-descriptions.shtml"


@dataclass(slots=True)
class MenuItem:
    menu: str  # "Analyze"
    item: str  # "Distribution"
    menu_path: str  # "Analyze > Distribution"
    description: str
    platform: str  # "windows" | "macos"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).replace(" ", " ").strip()


def parse_menu_page(html: str, platform: str = "windows") -> list[MenuItem]:
    """Parse one menu-descriptions page into flat menu items."""
    soup = BeautifulSoup(html, "lxml")
    content = soup.find("div", id="page_content") or soup

    items: list[MenuItem] = []
    for table in content.find_all("table"):
        caption = table.find("caption")
        if caption is None:
            continue
        menu_name = _clean(caption.get_text(" "))
        # "Analyze Menu" -> "Analyze"
        menu_name = re.sub(r"\s*Menu$", "", menu_name, flags=re.I)
        if not menu_name:
            continue

        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            item = _clean(cells[0].get_text(" "))
            desc = _clean(cells[1].get_text(" "))
            if not item or item.lower() == "item":
                continue
            items.append(
                MenuItem(
                    menu=menu_name,
                    item=item,
                    menu_path=f"{menu_name} > {item}",
                    description=desc,
                    platform=platform,
                )
            )

    return items


def build_menu_index(settings: Settings | None = None) -> list[MenuItem]:
    """Fetch and parse the menu reference pages."""
    st = settings or get_settings()
    items: list[MenuItem] = []

    for path, platform in ((WINDOWS_MENU_PAGE, "windows"), (MACOS_MENU_PAGE, "macos")):
        try:
            html = fetch_page(path, st)
        except Exception as exc:  # macOS page is optional
            if platform == "windows":
                raise
            print(f"  note: skipped {platform} menus ({type(exc).__name__})")
            continue
        items.extend(parse_menu_page(html, platform))

    return items


def save_menus(
    items: list[MenuItem], settings: Settings | None = None
) -> Path:
    st = settings or get_settings()
    st.ensure_dirs()
    out = st.knowledge_dir / "menus.json"
    out.write_text(
        json.dumps([asdict(i) for i in items], indent=2), encoding="utf-8"
    )
    return out


def load_menus(settings: Settings | None = None) -> list[MenuItem]:
    st = settings or get_settings()
    path = st.knowledge_dir / "menus.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [MenuItem(**r) for r in raw]


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    from collections import Counter

    st = get_settings()
    items = build_menu_index(st)
    by_platform = Counter(i.platform for i in items)
    by_menu = Counter(i.menu for i in items if i.platform == "windows")

    print(f"menu items: {len(items):,}  ({dict(by_platform)})")
    print("\nwindows menus:")
    for menu, n in by_menu.most_common():
        print(f"  {n:3d}  {menu}")

    print("\nsample (Analyze):")
    for i in items:
        if i.menu == "Analyze" and i.platform == "windows":
            print(f"  {i.menu_path:38s} {i.description[:64]}")

    print(f"\nwrote {save_menus(items, st)}")
