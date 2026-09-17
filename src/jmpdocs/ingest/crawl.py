"""Fetch JMP documentation pages, caching them on disk.

Pages are small (~12 KB) and static, so the crawl is cheap: ~3,900 requests at
concurrency 8 takes roughly 13 minutes cold. Everything lands in data/raw/,
which makes the crawl resumable -- an interrupted run picks up where it left
off, and re-parsing never re-downloads.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import httpx

from jmpdocs.config import Settings, get_settings


@dataclass(slots=True)
class FetchResult:
    path: str  # corpus-relative, e.g. "jmp/tabulate.shtml"
    ok: bool
    from_cache: bool
    status: int | None = None
    error: str | None = None


def cache_path_for(path: str, settings: Settings | None = None) -> Path:
    """Local cache location mirroring the site layout under data/raw/."""
    st = settings or get_settings()
    return st.raw_dir / path


def read_cached(path: str, settings: Settings | None = None) -> str | None:
    p = cache_path_for(path, settings)
    if p.exists() and p.stat().st_size > 0:
        return p.read_text(encoding="utf-8", errors="replace")
    return None


# --------------------------------------------------------------------------
# async crawl
# --------------------------------------------------------------------------


async def _fetch_one(
    client: httpx.AsyncClient,
    path: str,
    st: Settings,
    sem: asyncio.Semaphore,
) -> FetchResult:
    dest = cache_path_for(path, st)
    if dest.exists() and dest.stat().st_size > 0:
        return FetchResult(path=path, ok=True, from_cache=True)

    url = st.source.page_url(path)
    last_error: str | None = None
    last_status: int | None = None

    async with sem:
        for attempt in range(st.crawl.retries):
            try:
                if st.crawl.delay:
                    await asyncio.sleep(random.uniform(0, st.crawl.delay))
                resp = await client.get(url)
                last_status = resp.status_code
                if resp.status_code == 200:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(resp.text, encoding="utf-8")
                    return FetchResult(path=path, ok=True, from_cache=False, status=200)
                if resp.status_code == 404:
                    # A handful of TOC targets genuinely do not exist; don't retry.
                    return FetchResult(
                        path=path, ok=False, from_cache=False, status=404,
                        error="not found",
                    )
                last_error = f"HTTP {resp.status_code}"
            except Exception as exc:  # network hiccup, timeout, reset
                last_error = f"{type(exc).__name__}: {exc}"

            # exponential backoff with jitter
            await asyncio.sleep((2**attempt) * 0.5 + random.uniform(0, 0.3))

    return FetchResult(
        path=path, ok=False, from_cache=False, status=last_status, error=last_error
    )


async def crawl_pages_async(
    paths: Sequence[str],
    settings: Settings | None = None,
    progress: Callable[[int, int, FetchResult], None] | None = None,
) -> list[FetchResult]:
    st = settings or get_settings()
    st.ensure_dirs()

    sem = asyncio.Semaphore(st.crawl.concurrency)
    limits = httpx.Limits(
        max_connections=st.crawl.concurrency * 2,
        max_keepalive_connections=st.crawl.concurrency,
    )

    results: list[FetchResult] = []
    async with httpx.AsyncClient(
        timeout=st.crawl.timeout,
        headers={"User-Agent": st.crawl.user_agent},
        limits=limits,
        follow_redirects=True,
    ) as client:
        tasks = [
            asyncio.create_task(_fetch_one(client, p, st, sem)) for p in paths
        ]
        total = len(tasks)
        for done, coro in enumerate(asyncio.as_completed(tasks), start=1):
            res = await coro
            results.append(res)
            if progress is not None:
                progress(done, total, res)

    return results


def crawl_pages(
    paths: Iterable[str],
    settings: Settings | None = None,
    progress: Callable[[int, int, FetchResult], None] | None = None,
) -> list[FetchResult]:
    """Blocking wrapper around :func:`crawl_pages_async`."""
    return asyncio.run(crawl_pages_async(list(paths), settings, progress))


# --------------------------------------------------------------------------
# single-page helpers
# --------------------------------------------------------------------------


def fetch_page(
    path: str, settings: Settings | None = None, *, force: bool = False
) -> str:
    """Fetch (or read from cache) a single page's HTML."""
    st = settings or get_settings()
    if not force:
        cached = read_cached(path, st)
        if cached is not None:
            return cached

    st.ensure_dirs()
    resp = httpx.get(
        st.source.page_url(path),
        timeout=st.crawl.timeout,
        headers={"User-Agent": st.crawl.user_agent},
        follow_redirects=True,
    )
    resp.raise_for_status()
    dest = cache_path_for(path, st)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text, encoding="utf-8")
    return resp.text


def fetch_live(url: str, settings: Settings | None = None) -> str:
    """Fetch a URL straight from jmp.com, bypassing the cache entirely.

    Backs the `jmp_fetch_live` MCP tool, for when the index is stale or a page
    is newer than the last build.
    """
    st = settings or get_settings()
    if not url.startswith(st.source.base_url):
        raise ValueError(f"refusing to fetch outside {st.source.base_url}: {url}")
    resp = httpx.get(
        url,
        timeout=st.crawl.timeout,
        headers={"User-Agent": st.crawl.user_agent},
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.text
