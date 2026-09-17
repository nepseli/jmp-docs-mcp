"""Chunk the corpus and build the hybrid retrieval index.

Two indexes are built over the same chunks, because they fail in different
places:

  dense (FAISS)  BGE embeddings. Handles paraphrase and concept matching.
                 Unreliable on product proper nouns.

  lexical (BM25) Exact tokens. This is what actually resolves "Choose()" or
                 "Bootstrap Forest". Its text is *augmented* with the user-side
                 synonyms from the vocabulary bridge, which is what lets
                 "random forest" find a page that never says those words.

Every chunk also carries a contextual prefix (book > chapter > section) into the
embedding, because page titles repeat heavily across books -- "Responses"
appears half a dozen times -- and the bare text is often ambiguous without it.
"""

from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from jmpdocs.config import Settings, get_settings
from jmpdocs.ingest.parse import Document, Figure, read_corpus
from jmpdocs.knowledge.aliases import AliasBridge, get_alias_bridge

FAISS_FILE = "faiss.index"
BM25_FILE = "bm25.pkl"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_FIGURE_MARKER_RE = re.compile(r"\[FIGURE:\s*(.+?)\]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    path: str
    url: str
    title: str
    book: str
    breadcrumb: list[str]
    section: str
    page_type: str
    text: str
    n_chars: int
    figures: list[Figure] = field(default_factory=list)
    menu_paths: list[str] = field(default_factory=list)
    sample_data: list[str] = field(default_factory=list)
    alias_terms: list[str] = field(default_factory=list)
    ui_names: list[str] = field(default_factory=list)

    # ---- derived text views -------------------------------------------

    @property
    def context_line(self) -> str:
        trail = " > ".join(self.breadcrumb) if self.breadcrumb else self.book
        if self.section and self.section not in trail:
            trail = f"{trail} > {self.section}" if trail else self.section
        return trail

    def embed_text(self) -> str:
        """What the dense index sees: breadcrumb context, then the content."""
        return f"{self.context_line}\n{self.title}\n\n{self.text}"

    def bm25_text(self) -> str:
        """What the lexical index sees: content plus every alternative name."""
        parts = [
            self.title,
            self.context_line,
            self.text,
            " ".join(self.ui_names),
            " ".join(self.menu_paths),
            " ".join(self.sample_data),
            " ".join(self.alias_terms),
        ]
        return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split on ATX headings, returning (heading-trail, body) pairs."""
    lines = markdown.splitlines()
    sections: list[tuple[str, str]] = []
    trail: dict[int, str] = {}
    current: list[str] = []
    current_heading = ""

    def flush() -> None:
        body = "\n".join(current).strip()
        if body:
            sections.append((current_heading, body))
        current.clear()

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            level = len(m.group(1))
            text = m.group(2).strip()
            trail[level] = text
            for deeper in [k for k in trail if k > level]:
                trail.pop(deeper, None)
            current_heading = " > ".join(trail[k] for k in sorted(trail))
        else:
            current.append(line)
    flush()

    if not sections:
        body = markdown.strip()
        return [("", body)] if body else []
    return sections


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-aware splitter that never cuts inside a fenced code block."""
    if len(text) <= size:
        return [text]

    # keep fenced blocks atomic by treating them as single paragraphs
    parts: list[str] = []
    for block in re.split(r"(```.*?```)", text, flags=re.S):
        if not block:
            continue
        if block.startswith("```"):
            parts.append(block)
        else:
            parts.extend(p for p in re.split(r"\n\s*\n", block) if p.strip())

    chunks: list[str] = []
    buf: list[str] = []
    length = 0
    for p in parts:
        p_len = len(p) + 2
        if length + p_len > size and buf:
            chunks.append("\n\n".join(buf).strip())
            # carry a tail of the previous chunk for context continuity
            tail: list[str] = []
            carried = 0
            for prev in reversed(buf):
                if carried + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                carried += len(prev)
            buf = tail
            length = carried
        # a single oversized paragraph (a long code block) becomes its own chunk
        if p_len > size and not buf:
            chunks.append(p.strip())
            continue
        buf.append(p)
        length += p_len

    if buf:
        chunks.append("\n\n".join(buf).strip())
    return [c for c in chunks if c]


def chunk_document(
    doc: Document, st: Settings, bridge: AliasBridge
) -> list[Chunk]:
    figures_by_key = {
        (f.figure_id or f.caption or f.src).casefold(): f for f in doc.figures
    }

    pieces: list[tuple[str, str]] = []
    if doc.n_chars <= st.index.min_page_chars:
        pieces = [("", doc.markdown)] if doc.markdown.strip() else []
    else:
        for heading, body in _split_sections(doc.markdown):
            for piece in _split_long(body, st.index.chunk_size, st.index.chunk_overlap):
                pieces.append((heading, piece))

    chunks: list[Chunk] = []
    for i, (heading, text) in enumerate(pieces):
        # figures actually referenced inside this chunk
        figs: list[Figure] = []
        for marker in _FIGURE_MARKER_RE.findall(text):
            label = marker.strip()
            fig = figures_by_key.get(label.casefold())
            if fig is None:
                head = label.split(" - ")[0].casefold()
                fig = figures_by_key.get(head)
            if fig is None:
                for key, candidate in figures_by_key.items():
                    if key and (key in label.casefold() or label.casefold() in key):
                        fig = candidate
                        break
            if fig is not None and fig not in figs:
                figs.append(fig)

        menu_paths = [m for m in doc.menu_paths if m.split(" > ")[-1] in text] or (
            doc.menu_paths if i == 0 else []
        )
        samples = [s for s in doc.sample_data if s in text]

        chunk = Chunk(
            chunk_id=f"{doc.path}#{i}",
            path=doc.path,
            url=doc.url,
            title=doc.title,
            book=doc.book,
            breadcrumb=list(doc.breadcrumb),
            section=heading,
            page_type=doc.page_type,
            text=text,
            n_chars=len(text),
            figures=figs,
            menu_paths=menu_paths,
            sample_data=samples,
            ui_names=list(doc.ui_names),
        )
        # the vocabulary bridge, applied to title + text
        chunk.alias_terms = bridge.user_terms_for(f"{doc.title}\n{text}")
        chunks.append(chunk)

    return chunks


def chunk_corpus(
    docs: Iterable[Document] | None = None,
    settings: Settings | None = None,
    bridge: AliasBridge | None = None,
) -> list[Chunk]:
    st = settings or get_settings()
    br = bridge or get_alias_bridge()
    source = docs if docs is not None else read_corpus(st)
    out: list[Chunk] = []
    for doc in source:
        out.extend(chunk_document(doc, st, br))
    return out


# --------------------------------------------------------------------------
# index building
# --------------------------------------------------------------------------


def embed_texts(
    texts: Sequence[str],
    settings: Settings | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    st = settings or get_settings()
    model = SentenceTransformer(st.index.embed_model, device="cpu")
    vecs = model.encode(
        list(texts),
        batch_size=st.index.embed_batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # cosine similarity via inner product
        convert_to_numpy=True,
    )
    return vecs.astype("float32")


def build_index(
    chunks: Sequence[Chunk],
    settings: Settings | None = None,
    show_progress: bool = True,
) -> dict:
    import faiss
    from rank_bm25 import BM25Okapi

    st = settings or get_settings()
    st.ensure_dirs()
    started = time.time()

    # ---- dense ----------------------------------------------------------
    vectors = embed_texts([c.embed_text() for c in chunks], st, show_progress)
    dim = int(vectors.shape[1])
    # exact search: at ~10k vectors an approximate index buys nothing
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(st.index_dir / FAISS_FILE))

    # ---- lexical --------------------------------------------------------
    corpus_tokens = [tokenize(c.bm25_text()) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    with (st.index_dir / BM25_FILE).open("wb") as fh:
        pickle.dump(bm25, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- chunk store ----------------------------------------------------
    with (st.index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    manifest = {
        "jmp_version": st.source.version,
        "base_url": st.source.base_url,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(time.time() - started, 1),
        "embed_model": st.index.embed_model,
        "rerank_model": st.index.rerank_model,
        "embed_dim": dim,
        "n_chunks": len(chunks),
        "n_pages": len({c.path for c in chunks}),
        "n_books": len({c.book for c in chunks if c.book}),
        "n_figures": len({f.src for c in chunks for f in c.figures}),
        "chunk_size": st.index.chunk_size,
        "chunk_overlap": st.index.chunk_overlap,
        "alias_entries": len(get_alias_bridge()),
    }
    (st.index_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_manifest(settings: Settings | None = None) -> dict:
    st = settings or get_settings()
    p = st.index_dir / MANIFEST_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_chunks(settings: Settings | None = None) -> list[Chunk]:
    st = settings or get_settings()
    path = st.index_dir / CHUNKS_FILE
    if not path.exists():
        return []
    out: list[Chunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            raw["figures"] = [Figure(**f) for f in raw.get("figures", [])]
            out.append(Chunk(**raw))
    return out
