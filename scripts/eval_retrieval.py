"""Score retrieval against the golden question set.

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --no-rerank      # measure the rerank lift
    python scripts/eval_retrieval.py --no-alias       # measure the bridge lift
    python scripts/eval_retrieval.py --category vocabulary_bridge

Reports recall@k and MRR overall and per category, then lists every miss so
tuning is driven by evidence rather than vibes.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jmpdocs.config import get_settings  # noqa: E402
from jmpdocs.knowledge.aliases import AliasBridge  # noqa: E402
from jmpdocs.knowledge.intent import classify_intent  # noqa: E402
from jmpdocs.retrieval.hybrid import search  # noqa: E402
from jmpdocs.retrieval.store import get_store  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden_questions.yaml"


def rank_of_first_hit(hits, expected: list[str]) -> int | None:
    wanted = [e.casefold() for e in expected]
    for i, h in enumerate(hits, start=1):
        title = h.chunk.title.casefold()
        if any(w in title for w in wanted):
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--rerank", action="store_true", help="force reranking on")
    ap.add_argument("--no-alias", action="store_true")
    ap.add_argument("--category", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    st = get_settings()
    store = get_store()
    bridge = AliasBridge(aliases=[]) if args.no_alias else None

    spec = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.category:
        questions = [q for q in questions if q.get("category") == args.category]

    print(f"index    : {len(store):,} chunks, {len(store.paths):,} pages")
    rerank = True if args.rerank else (False if args.no_rerank else st.retrieval.rerank)
    print(f"settings : k={args.k} rerank={rerank} alias={not args.no_alias}")
    print(f"questions: {len(questions)}\n")

    per_cat: dict[str, list[int | None]] = defaultdict(list)
    intent_ok = intent_total = 0
    misses = []
    started = time.time()

    for i, item in enumerate(questions, start=1):
        res = search(
            item["q"],
            k=args.k,
            rerank=rerank,
            store=store,
            settings=st,
            bridge=bridge,
        )
        rank = rank_of_first_hit(res.hits, item.get("expect_title_any", []))
        per_cat[item.get("category", "other")].append(rank)

        if "expect_intent" in item:
            intent_total += 1
            got = classify_intent(item["q"])
            if got == item["expect_intent"]:
                intent_ok += 1
            elif not args.quiet:
                print(f"  intent  {item['id']}: expected {item['expect_intent']}, got {got}")

        if rank is None:
            misses.append((item, res))

        if not args.quiet:
            mark = f"@{rank}" if rank else "MISS"
            print(f"  [{i:2d}/{len(questions)}] {mark:>5}  {item['id']:22s} {item['q'][:52]}")

    elapsed = time.time() - started
    all_ranks = [r for ranks in per_cat.values() for r in ranks]
    found = [r for r in all_ranks if r]
    recall = len(found) / len(all_ranks) if all_ranks else 0
    mrr = sum(1 / r for r in found) / len(all_ranks) if all_ranks else 0

    print("\n" + "=" * 62)
    print(f"RECALL@{args.k}: {recall:.1%}   MRR: {mrr:.3f}   "
          f"({len(found)}/{len(all_ranks)})   {elapsed/len(all_ranks):.2f}s/query")
    if intent_total:
        print(f"INTENT ACCURACY: {intent_ok}/{intent_total} ({intent_ok/intent_total:.0%})")
    print("=" * 62)

    print("\nper category:")
    for cat in sorted(per_cat):
        ranks = per_cat[cat]
        hit = [r for r in ranks if r]
        c_mrr = sum(1 / r for r in hit) / len(ranks) if ranks else 0
        print(f"  {cat:20s} recall {len(hit):2d}/{len(ranks):<2d} ({len(hit)/len(ranks):5.0%})  mrr {c_mrr:.3f}")

    if misses:
        print(f"\n{len(misses)} miss(es):")
        for item, res in misses:
            print(f"\n  {item['id']}: {item['q']}")
            print(f"    expected title containing: {item.get('expect_title_any')}")
            print(f"    intent={res.intent}  aliases={res.alias_terms}")
            for h in res.hits[:4]:
                print(f"      - {h.chunk.title[:52]:54s} [{h.chunk.page_type}] {h.score:.3f}")

    return 0 if recall >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
