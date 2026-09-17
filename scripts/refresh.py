"""Refresh the corpus and index against the live JMP documentation.

    python scripts/refresh.py --check     # report staleness only
    python scripts/refresh.py             # re-crawl changed pages, rebuild index
    python scripts/refresh.py --full      # discard the cache and rebuild everything

The crawler is cache-backed, so a routine refresh only pays for pages that
actually changed. Use --full after a JMP version bump (change source.version
and source.base_url in config.yaml first).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jmpdocs.config import get_settings  # noqa: E402
from jmpdocs.ingest.build_index import load_manifest  # noqa: E402
from jmpdocs.ingest.crawl import fetch_page  # noqa: E402
from jmpdocs.ingest.parse import parse_page  # noqa: E402

PROBE_PAGE = "jmp/tabulate.shtml"


def check(st) -> int:
    manifest = load_manifest(st)
    if not manifest:
        print("no index built yet")
        return 1

    print(f"indexed JMP version : {manifest.get('jmp_version')}")
    print(f"index built         : {manifest.get('built_at')}")
    print(f"chunks / pages      : {manifest.get('n_chunks'):,} / {manifest.get('n_pages'):,}")

    try:
        html = fetch_page(PROBE_PAGE, st, force=True)
        live = parse_page(html, PROBE_PAGE, None, st)
        print(f"live publication    : {live.published}")
    except Exception as exc:
        print(f"could not reach jmp.com: {type(exc).__name__}: {exc}")
        return 1
    return 0


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report staleness and exit")
    ap.add_argument("--full", action="store_true", help="discard caches first")
    args = ap.parse_args()

    st = get_settings()
    if args.check:
        return check(st)

    if args.full:
        for d in (st.raw_dir, st.index_dir):
            if d.exists():
                print(f"removing {d}")
                shutil.rmtree(d)
        st.ensure_dirs()

    python = sys.executable
    rc = run([python, str(ROOT / "scripts" / "build_corpus.py")])
    if rc:
        return rc
    return run([python, str(ROOT / "scripts" / "build_index.py")])


if __name__ == "__main__":
    raise SystemExit(main())
