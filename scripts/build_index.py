"""Chunk the corpus and build the FAISS + BM25 hybrid index.

    python scripts/build_index.py

Assumes scripts/build_corpus.py has already run. Embedding ~10k chunks on CPU
takes roughly 10-15 minutes the first time; the model weights (~440 MB) are
downloaded once and cached by sentence-transformers.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jmpdocs.config import get_settings  # noqa: E402
from jmpdocs.ingest.build_index import build_index, chunk_corpus  # noqa: E402
from jmpdocs.knowledge.aliases import get_alias_bridge  # noqa: E402


def main() -> int:
    st = get_settings()
    if not st.corpus_path.exists():
        print(f"no corpus at {st.corpus_path}\nRun: python scripts/build_corpus.py")
        return 1

    bridge = get_alias_bridge()
    print(f"data dir      : {st.data_dir}")
    print(f"embed model   : {st.index.embed_model}")
    print(f"alias entries : {len(bridge)}")

    print("\n[1/2] chunking corpus")
    started = time.time()
    chunks = chunk_corpus(settings=st, bridge=bridge)
    pages = len({c.path for c in chunks})
    chars = sum(c.n_chars for c in chunks)
    with_alias = sum(1 for c in chunks if c.alias_terms)
    print(f"  {len(chunks):,} chunks from {pages:,} pages  ({time.time()-started:.1f}s)")
    print(f"  mean chunk    : {chars//max(len(chunks),1):,} chars")
    print(f"  chunks w/ alias terms : {with_alias:,} ({with_alias/max(len(chunks),1)*100:.0f}%)")
    print(f"  chunks w/ figures     : {sum(1 for c in chunks if c.figures):,}")

    print("\n  chunks per book:")
    for book, n in Counter(c.book for c in chunks).most_common():
        print(f"    {n:6,}  {book or '(none)'}")

    print("\n[2/2] embedding + indexing")
    manifest = build_index(chunks, st, show_progress=True)

    print("\n" + "=" * 60)
    print("INDEX BUILT")
    print("=" * 60)
    for key in (
        "n_chunks", "n_pages", "n_books", "n_figures", "embed_model",
        "embed_dim", "alias_entries", "build_seconds", "built_at",
    ):
        print(f"  {key:15s}: {manifest.get(key)}")

    size = sum(f.stat().st_size for f in st.index_dir.iterdir() if f.is_file())
    print(f"  {'index size':15s}: {size/1e6:.1f} MB")
    print(f"\n  -> {st.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
