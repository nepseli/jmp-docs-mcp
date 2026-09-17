"""Turn a JMP help page into clean Markdown plus structured metadata.

The source is WebWorks ePublisher output, which is verbose but *consistent*:
meaning is carried in CSS classes rather than semantic tags. Everything below
is driven by the class vocabulary actually observed across a stratified sample
of the corpus, not guesswork. The main families are:

    body text     p.body, p.bodyKeepWithNext, p.indent1
    lists         p.N1bullet / p.N2bullet, p.number
    definitions   p.defTerm / p.defText (+Indent) -- how red-triangle *options*
                  are documented, so these matter a lot for option lookups
    JSL syntax    p.S1SynObj, p.S2Syn, p.S3Syn, p.S1alt2, pre.code, p.codeOutput
    JSL arguments p.S3Arg (name) / p.S3ArgText (description)
    inline code   span.code, span.command, span.argument, span.JSL*, span.colName
    figures       p.Figure_Title (+ span.zmpFigTab) then p.figureContainer > img
    navigation    anything whose class contains "TOC", plus h2.contents

That last family is dropped on purpose. Chapter landing pages embed 3-19
auto-generated tables of contents; left in the index they become link-list
chunks that match every query and answer none of them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import quote, unquote

from bs4 import BeautifulSoup, NavigableString, Tag

from jmpdocs.config import Settings, get_settings
from jmpdocs.ingest.crawl import read_cached
from jmpdocs.ingest.toc import PageMeta, classify_page_type

# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Figure:
    figure_id: str  # "Figure 9.1" ("" when the image has no numbered caption)
    caption: str  # "Tabulate Examples"
    src: str  # corpus-relative, e.g. "jmp/images/1-103.png"
    alt: str


@dataclass(slots=True)
class Document:
    path: str
    url: str
    title: str
    book: str
    breadcrumb: list[str]
    page_type: str
    published: str
    markdown: str
    n_chars: int
    figures: list[Figure] = field(default_factory=list)
    menu_paths: list[str] = field(default_factory=list)
    sample_data: list[str] = field(default_factory=list)
    ui_names: list[str] = field(default_factory=list)
    prev: str = ""
    next: str = ""


# --------------------------------------------------------------------------
# extraction helpers
# --------------------------------------------------------------------------

JMP_MENUS = (
    "File", "Edit", "Tables", "Rows", "Cols", "DOE", "Analyze", "Graph",
    "Tools", "Project", "View", "Window", "Help", "Format",
)

_MENU_RE = re.compile(
    rf"\b(?P<root>{'|'.join(JMP_MENUS)})\s*>\s*(?P<rest>[^\n]{{1,160}})"
)
_SAMPLE_DATA_RE = re.compile(r"([A-Za-z0-9 _&'.-]{1,60}?\.jmp)\b")

# Menu items are Title Case but contain lowercase connectives ("Fit Y by X",
# "Save Script to Data Table"). Prose that follows a menu path does not resume
# in Title Case, so: accept a lowercase word only when a capitalised word
# follows it. That keeps "Fit Y by X" whole while cutting "and open ..." off
# the end of "Help > Sample Data Folder and open the file".
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9()'/&_-]*")

# Leading prose that regularly runs into a sample-data filename.
_SAMPLE_PREFIX_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "open", "opens",
    "select", "use", "uses", "using", "see", "in", "from", "and", "or",
    "figure", "table", "example", "shows", "show", "data", "folder",
    "sample", "named", "called", "file",
}


def _trim_menu_segment(segment: str) -> str:
    """Cut a captured menu segment back to the menu item itself."""
    segment = segment.strip()
    # stop at sentence-ish punctuation
    segment = re.split(r"[.;:,)\]]\s|[.;:]\s*$|\s+\d+\.\s", segment)[0]
    words = _WORD_RE.findall(segment)
    kept: list[str] = []
    for i, w in enumerate(words):
        if w[0].isupper() or w[0].isdigit():
            kept.append(w)
            continue
        # lowercase word: keep only if a capitalised word follows
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if kept and nxt and (nxt[0].isupper() or nxt[0].isdigit()):
            kept.append(w)
            continue
        break
    return " ".join(kept).strip()


def _trim_sample_name(raw: str) -> str:
    """Cut prose off the front of a matched '<something>.jmp' run."""
    stem = raw[: -len(".jmp")]
    words = stem.split()
    kept: list[str] = []
    for w in reversed(words):
        # Digits are excluded on purpose: numbered steps ("1. 3. In Foo.jmp")
        # otherwise leak into the filename.
        if w[0].isupper() or "_" in w:
            kept.append(w)
        else:
            break
    kept.reverse()
    while kept and kept[0].casefold() in _SAMPLE_PREFIX_STOPWORDS:
        kept.pop(0)
    return (" ".join(kept) + ".jmp") if kept else ""

_INLINE_CODE_CLASSES = {
    "code", "command", "argument", "colName",
    "JSLNumber", "JSLString", "JSLOperatorName", "EquationVariables",
}
_CODE_BLOCK_CLASSES = {"code", "codeOutput", "S1SynObj", "S2Syn", "S3Syn", "S1alt2"}
_BULLET_CLASSES = {"N1bullet", "N2bullet", "N3bullet", "bullet"}
_TERM_CLASSES = {"defTerm", "defTermIndent"}
_TERM_TEXT_CLASSES = {"defText", "defTextIndent"}
_DROP_CLASSES = {"ww_skin", "ww_skin_dropdown_arrow", "fa", "zmpFigTab"}


def _classes(el: Tag) -> set[str]:
    attrs = getattr(el, "attrs", None)
    if not attrs:
        return set()
    value = attrs.get("class") or ()
    if isinstance(value, str):
        value = value.split()
    return set(value)


def _is_nav(el: Tag) -> bool:
    """Auto-generated in-page tables of contents, and their 'Contents' heading."""
    if getattr(el, "decomposed", False):
        return False
    cls = _classes(el)
    if any("TOC" in c for c in cls):
        return True
    if "contents" in cls and el.name in {"h1", "h2", "h3"}:
        return True
    return False


def _norm_ws(text: str) -> str:
    return re.sub(r"[ \t ]+", " ", text).strip()


def _normalize_img_src(src: str) -> str:
    """'../jmp/images/paste%20new.png' -> 'jmp/images/paste new.png'."""
    src = (src or "").strip().replace("\\", "/")
    src = re.sub(r"^(\.\./)+", "", src)
    src = src.lstrip("/")
    # a few srcs are percent-encoded; the file on the server is not
    return unquote(src)


# --------------------------------------------------------------------------
# inline rendering
# --------------------------------------------------------------------------


def _render_inline(node: Tag | NavigableString) -> str:
    """Render an element's children to inline Markdown."""
    if isinstance(node, NavigableString):
        return str(node)

    out: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            out.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue

        cls = _classes(child)
        if child.name in {"i", "svg"} and "fa" in cls:
            continue  # font-awesome chrome
        if cls & _DROP_CLASSES and child.name == "span" and not child.get_text(strip=True):
            continue

        if child.name == "img":
            continue  # handled at block level

        text = _render_inline(child)

        if child.name in {"b", "strong"}:
            out.append(f"**{_norm_ws(text)}**" if text.strip() else "")
        elif child.name in {"i", "em"}:
            out.append(f"*{_norm_ws(text)}*" if text.strip() else "")
        elif child.name in {"code", "tt"}:
            out.append(f"`{_norm_ws(text)}`" if text.strip() else "")
        elif child.name == "span" and cls & _INLINE_CODE_CLASSES:
            t = _norm_ws(text)
            out.append(f"`{t}`" if t else "")
        elif child.name == "span" and "BookTitle" in cls:
            t = _norm_ws(text)
            out.append(f"*{t}*" if t else "")
        else:
            out.append(text)

    return "".join(out)


def _inline_text(el: Tag) -> str:
    return _norm_ws(_render_inline(el))


# --------------------------------------------------------------------------
# table rendering
# --------------------------------------------------------------------------


def _render_table(table: Tag) -> str:
    caption_el = table.find("caption")
    caption = _inline_text(caption_el) if caption_el else ""

    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False) or tr.find_all(["th", "td"])
        if not cells:
            continue
        rows.append([_inline_text(c).replace("|", "\\|") for c in cells])

    if not rows:
        return f"**{caption}**" if caption else ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    header, body = rows[0], rows[1:]
    lines = []
    if caption:
        lines.append(f"**{caption}**")
        lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * width) + "|")
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# block rendering
# --------------------------------------------------------------------------


class _Renderer:
    def __init__(self) -> None:
        self.blocks: list[str] = []
        self.figures: list[Figure] = []
        self._pending_fig: tuple[str, str] | None = None  # (figure_id, caption)
        self._code_buf: list[str] = []

    # -- code buffering so consecutive syntax lines form one fence ----------

    def _flush_code(self) -> None:
        if self._code_buf:
            body = "\n".join(self._code_buf).rstrip()
            if body:
                self.blocks.append(f"```jsl\n{body}\n```")
            self._code_buf.clear()

    def _push(self, text: str) -> None:
        self._flush_code()
        if text:
            self.blocks.append(text)

    def _push_code(self, text: str) -> None:
        if text:
            self._code_buf.append(text)

    # -- figures -----------------------------------------------------------

    def _handle_figure_title(self, el: Tag) -> None:
        tab = el.find("span", class_="zmpFigTab")
        figure_id = _norm_ws(tab.get_text(" ")) if tab else ""
        if tab:
            tab.extract()
        caption = _inline_text(el)
        self._pending_fig = (figure_id, caption)

    def _handle_img(self, img: Tag) -> None:
        src = _normalize_img_src(img.get("src", ""))
        if not src:
            return
        alt = _norm_ws(img.get("alt") or img.get("title") or "")
        figure_id, caption = self._pending_fig or ("", alt)
        self._pending_fig = None

        self.figures.append(
            Figure(figure_id=figure_id, caption=caption or alt, src=src, alt=alt)
        )
        # An inline anchor keeps the figure findable by caption in the text index
        # and lets the UI render the actual image next to the cited passage.
        label = " - ".join(p for p in (figure_id, caption or alt) if p)
        self._push(f"[FIGURE: {label or src}]")

    # -- main walk ---------------------------------------------------------

    def walk(self, container: Tag) -> None:
        for el in container.children:
            if isinstance(el, NavigableString):
                continue
            if not isinstance(el, Tag):
                continue
            self._element(el)
        self._flush_code()

    def _element(self, el: Tag) -> None:
        if _is_nav(el):
            return

        cls = _classes(el)
        name = el.name

        if name in {"script", "style", "noscript"}:
            return

        # containers we descend into
        if name in {"div", "section", "article"}:
            self.walk(el)
            return

        if name == "table":
            self._push(_render_table(el))
            return

        if name in {"ul", "ol"}:
            ordered = name == "ol"
            items = []
            for i, li in enumerate(el.find_all("li", recursive=False), start=1):
                t = _inline_text(li)
                if t:
                    items.append(f"{i}. {t}" if ordered else f"- {t}")
            self._push("\n".join(items))
            return

        if name == "img":
            self._handle_img(el)
            return

        if re.fullmatch(r"h[1-6]", name):
            level = int(name[1])
            text = _inline_text(el)
            if text:
                self._push(f"{'#' * min(level, 6)} {text}")
            return

        if name == "pre":
            self._flush_code()
            body = el.get_text("\n").strip("\n")
            if body.strip():
                self.blocks.append(f"```jsl\n{body}\n```")
            return

        if name == "p":
            self._paragraph(el, cls)
            return

        # anything else: descend if it has element children, else emit text
        if el.find(True) is not None:
            self.walk(el)
        else:
            self._push(_inline_text(el))

    def _paragraph(self, el: Tag, cls: set[str]) -> None:
        # figure caption / image container
        if "Figure_Title" in cls:
            self._handle_figure_title(el)
            return
        if "figureContainer" in cls:
            for img in el.find_all("img"):
                self._handle_img(img)
            return

        img = el.find("img")
        if img is not None and not el.get_text(strip=True):
            self._handle_img(img)
            return

        text = _inline_text(el)
        if not text:
            return

        # JSL syntax and code output accumulate into a single fenced block
        if cls & _CODE_BLOCK_CLASSES:
            raw = _norm_ws(el.get_text(" "))
            self._push_code(raw)
            return

        if cls & _TERM_CLASSES:
            self._push(f"**{text}**")
            return
        if cls & _TERM_TEXT_CLASSES:
            self._push(text)
            return
        if "S3Arg" in cls:
            self._push(f"**{text}**")
            return
        if "S3ArgText" in cls:
            self._push(text)
            return
        if cls & _BULLET_CLASSES:
            depth = 0
            for c in cls:
                m = re.fullmatch(r"N(\d)bullet", c)
                if m:
                    depth = int(m.group(1)) - 1
            self._push(f"{'  ' * depth}- {text}")
            return
        if "number" in cls:
            self._push(f"1. {text}")
            return
        if "Note" in cls or el.find("span", class_="Note") is not None:
            # the source usually already opens with "Note:"; don't double it up
            body = re.sub(r"^\s*(note|tip|caution|warning)\s*:\s*", "", text, flags=re.I)
            self._push(f"> **Note:** {body}")
            return

        self._push(text)


# --------------------------------------------------------------------------
# page-level parsing
# --------------------------------------------------------------------------


_FENCE_RE = re.compile(r"^```jsl\n(.*)\n```$", re.S)


def _collapse_blocks(blocks: Iterable[str]) -> str:
    out: list[str] = []
    for b in blocks:
        b = b.rstrip()
        if not b:
            continue

        # Multi-line JSL examples arrive as one <pre> per line. Merge adjacent
        # fences so a script reads as a single runnable block rather than a
        # stack of one-line snippets.
        if out:
            prev_fence = _FENCE_RE.match(out[-1])
            this_fence = _FENCE_RE.match(b)
            if prev_fence and this_fence:
                out[-1] = f"```jsl\n{prev_fence.group(1)}\n{this_fence.group(1)}\n```"
                continue

            # merge consecutive list items into one block
            bullets = ("- ", "  - ", "1. ")
            if b.startswith(bullets) and out[-1].startswith(bullets):
                out[-1] = out[-1] + "\n" + b
                continue

        out.append(b)

    text = "\n\n".join(out).strip()
    # The source marks up the name but not the parens: `Match`() -> `Match()`.
    # Keeping the identifier intact matters for BM25, which is what actually
    # resolves exact JSL function lookups.
    text = re.sub(r"`([A-Za-z][\w ]*)`\s*\(\s*\)", r"`\1()`", text)
    return text


def _extract_menu_paths(text: str) -> list[str]:
    found: list[str] = []
    for m in _MENU_RE.finditer(text):
        segments = [_trim_menu_segment(s) for s in m.group("rest").split(">")]
        kept: list[str] = []
        for seg in segments:
            if not seg:
                break
            kept.append(seg)
        if not kept:
            continue
        path = " > ".join([m.group("root"), *kept])
        if path not in found:
            found.append(path)
    return found


def _extract_sample_data(text: str) -> list[str]:
    found: list[str] = []
    for m in _SAMPLE_DATA_RE.finditer(text):
        name = _trim_sample_name(_norm_ws(m.group(1)))
        if name and name not in found:
            found.append(name)
    return found


def parse_page(
    html: str,
    path: str,
    meta: PageMeta | None = None,
    settings: Settings | None = None,
) -> Document:
    """Parse one cached help page into a :class:`Document`."""
    st = settings or get_settings()
    soup = BeautifulSoup(html, "lxml")

    # breadcrumb -> book + chapter trail
    breadcrumb: list[str] = []
    crumb_el = soup.find("div", class_="ww_skin_breadcrumbs")
    if crumb_el is not None:
        raw = crumb_el.get_text(">", strip=True)
        breadcrumb = [_norm_ws(p) for p in raw.split(">") if _norm_ws(p)]

    published = ""
    date_el = soup.find("div", class_="ww_creation_date")
    if date_el is not None:
        published = _norm_ws(date_el.get_text(" ")).replace("Publication date:", "").strip()

    def _rel(rel: str) -> str:
        link = soup.find("link", rel=rel)
        href = (link.get("href") if link else "") or ""
        href = href.split("#")[0].strip()
        return f"{st.source.book_root}/{href}" if href else ""

    content = soup.find("div", id="page_content")
    if content is None:
        content = soup.find("div", id="page_content_container") or soup.body or soup

    # drop chrome that carries no documentation value
    for sel_id in ("community-message", "page_dates"):
        node = content.find(id=sel_id)
        if node is not None:
            node.decompose()
    for node in content.find_all("footer"):
        node.decompose()
    for node in list(content.find_all(True)):
        # decompose() invalidates descendants, so re-check before touching each
        if getattr(node, "decomposed", False):
            continue
        if _is_nav(node):
            node.decompose()

    title_el = content.find(re.compile(r"^h[1-3]$"))
    title = _inline_text(title_el) if title_el else ""
    if not title and meta is not None:
        title = meta.title

    renderer = _Renderer()
    renderer.walk(content)
    markdown = _collapse_blocks(renderer.blocks)

    # Keep ">" here: menu paths ("Analyze > Distribution") depend on it.
    # Only Markdown emphasis/code/heading markers are stripped.
    plain = re.sub(r"[`*]", "", markdown)
    plain = re.sub(r"(?m)^#{1,6}\s*", "", plain)
    plain = re.sub(r"(?m)^>\s*\*{0,2}Note:\*{0,2}\s*", "Note: ", plain)
    book = breadcrumb[0] if breadcrumb else ""
    page_type = meta.page_type if meta is not None else classify_page_type(title)

    return Document(
        path=path,
        url=st.source.page_url(path),
        title=title,
        book=book,
        breadcrumb=breadcrumb,
        page_type=page_type,
        published=published,
        markdown=markdown,
        n_chars=len(markdown),
        figures=renderer.figures,
        menu_paths=_extract_menu_paths(plain),
        sample_data=_extract_sample_data(plain),
        ui_names=list(meta.ui_names) if meta is not None else [],
        prev=_rel("Prev"),
        next=_rel("Next"),
    )


def parse_cached_page(
    path: str, meta: PageMeta | None = None, settings: Settings | None = None
) -> Document | None:
    st = settings or get_settings()
    html = read_cached(path, st)
    if html is None:
        return None
    return parse_page(html, path, meta, st)


# --------------------------------------------------------------------------
# corpus persistence
# --------------------------------------------------------------------------


def write_corpus(docs: Iterable[Document], settings: Settings | None = None) -> Path:
    st = settings or get_settings()
    st.ensure_dirs()
    out = st.corpus_path
    with out.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    return out


def read_corpus(settings: Settings | None = None) -> Iterator[Document]:
    st = settings or get_settings()
    if not st.corpus_path.exists():
        return
    with st.corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            raw["figures"] = [Figure(**f) for f in raw.get("figures", [])]
            yield Document(**raw)
