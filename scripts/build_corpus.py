"""Build the JMP documentation corpus: TOC -> crawl -> parse -> images.

Everything is cached on disk, so re-running is cheap and an interrupted run
resumes where it stopped.

    python scripts/build_corpus.py            # full build
    python scripts/build_corpus.py --limit 50 # quick smoke test
    python scripts/build_corpus.py --no-images
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jmpdocs.config import get_settings  # noqa: E402
from jmpdocs.ingest.crawl import crawl_pages  # noqa: E402
from jmpdocs.ingest.images import download_images  # noqa: E402
from jmpdocs.ingest.menus import build_menu_index, save_menus  # noqa: E402
from jmpdocs.ingest.parse import parse_cached_page, write_corpus  # noqa: E402
from jmpdocs.ingest.toc import load_master_toc, save_knowledge  # noqa: E402


def _bar(done: int, total: int, label: str, started: float) -> None:
    if done % 100 and done != total:
        return
    pct = done / total * 100
    elapsed = time.time() - started
    rate = done / elapsed if elapsed else 0
    eta = (total - done) / rate if rate else 0
    sys.stdout.write(
        f"\r  {label}: {done:>5}/{total} ({pct:5.1f}%)  "
        f"{rate:5.1f}/s  eta {eta/60:4.1f}m   "
    )
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process the first N pages")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--refresh-toc", action="store_true", help="re-download the master TOC")
    args = ap.parse_args()

    st = get_settings()
    st.ensure_dirs()
    print(f"data dir: {st.data_dir}\n")

    # ---- 1. table of contents ------------------------------------------
    print("[1/5] master table of contents")
    pages, ui_map, toc = load_master_toc(st, force=args.refresh_toc)
    p_path, u_path = save_knowledge(pages, ui_map, st)
    print(f"  {len(toc):,} TOC entries -> {len(pages):,} unique pages")
    print(f"  {len(ui_map):,} UI-element mappings")

    # ---- 2. menu reference ---------------------------------------------
    print("\n[2/5] menu reference")
    menus = build_menu_index(st)
    save_menus(menus, st)
    print(f"  {len(menus):,} menu items across {len({m.menu for m in menus})} menus")

    # ---- 3. crawl -------------------------------------------------------
    paths = sorted(pages)
    if args.limit:
        paths = paths[: args.limit]
    print(f"\n[3/5] crawling {len(paths):,} pages (concurrency {st.crawl.concurrency})")
    started = time.time()
    results = crawl_pages(paths, st, lambda d, t, _r: _bar(d, t, "fetch", started))
    ok = [r for r in results if r.ok]
    cached = [r for r in ok if r.from_cache]
    failed = [r for r in results if not r.ok]
    print(f"  ok={len(ok):,}  (cached={len(cached):,})  failed={len(failed)}")
    for r in failed[:10]:
        print(f"    FAIL {r.path}  status={r.status}  {r.error}")

    # ---- 4. parse -------------------------------------------------------
    print(f"\n[4/5] parsing")
    started = time.time()
    docs = []
    for i, path in enumerate(paths, start=1):
        doc = parse_cached_page(path, pages.get(path), st)
        if doc is not None:
            docs.append(doc)
        _bar(i, len(paths), "parse", started)

    corpus_path = write_corpus(docs, st)
    print(f"  {len(docs):,} documents -> {corpus_path}")

    # ---- 5. images ------------------------------------------------------
    srcs = sorted({f.src for d in docs for f in d.figures})
    if args.no_images:
        print(f"\n[5/5] images: skipped ({len(srcs):,} referenced)")
    else:
        print(f"\n[5/5] downloading {len(srcs):,} images")
        started = time.time()
        img_results = download_images(srcs, st, lambda d, t, _r: _bar(d, t, "image", started))
        img_ok = [r for r in img_results if r.ok]
        img_bad = [r for r in img_results if not r.ok]
        total_mb = sum(r.n_bytes for r in img_ok) / 1e6
        print(f"  ok={len(img_ok):,}  failed={len(img_bad)}  ({total_mb:.0f} MB)")
        for r in img_bad[:10]:
            print(f"    FAIL {r.src}  {r.error}")

    # ---- report ---------------------------------------------------------
    print("\n" + "=" * 62)
    print("CORPUS SUMMARY")
    print("=" * 62)
    total_chars = sum(d.n_chars for d in docs)
    print(f"  documents        : {len(docs):,}")
    print(f"  total text       : {total_chars/1e6:.1f} M chars  (~{total_chars/4/1e6:.1f} M tokens)")
    print(f"  mean page        : {total_chars//max(len(docs),1):,} chars")
    print(f"  figures          : {sum(len(d.figures) for d in docs):,}")
    print(f"  menu-path refs   : {sum(len(d.menu_paths) for d in docs):,}")
    print(f"  sample-data refs : {len({s for d in docs for s in d.sample_data}):,} distinct")

    print("\n  books:")
    for book, n in Counter(d.book for d in docs).most_common():
        print(f"    {n:5d}  {book or '(none)'}")

    print("\n  page types:")
    for t, n in Counter(d.page_type for d in docs).most_common():
        print(f"    {n:5d}  {t}")

    # ---- assertions -----------------------------------------------------
    print("\n  health checks:")
    problems = []

    def check(label: str, bad: int, tolerance: int = 0) -> None:
        status = "ok " if bad <= tolerance else "FAIL"
        if bad > tolerance:
            problems.append(label)
        print(f"    [{status}] {label}: {bad}")

    check("pages that failed to fetch", len(failed), tolerance=25)
    check("documents with empty text", sum(1 for d in docs if d.n_chars < 40), tolerance=40)
    check("documents missing a title", sum(1 for d in docs if not d.title))
    check("documents missing a book", sum(1 for d in docs if not d.book), tolerance=25)
    check("TOC navigation residue", sum(1 for d in docs if "ChapterTOC" in d.markdown))

    if problems:
        print(f"\n  {len(problems)} check(s) failed")
        return 1
    print("\n  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
