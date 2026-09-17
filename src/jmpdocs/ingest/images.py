"""Download the figure images referenced by parsed pages.

Roughly 3,200 PNGs, ~40 KB each. They are never sent to a model -- per the
project's display-only choice they exist so the chat UI can show the actual
JMP dialog next to the passage that cites it.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.parse import quote

import httpx

from jmpdocs.config import Settings, get_settings


@dataclass(slots=True)
class ImageResult:
    src: str
    ok: bool
    from_cache: bool
    n_bytes: int = 0
    error: str | None = None


def image_path_for(src: str, settings: Settings | None = None) -> Path:
    """Local path for a corpus-relative image src such as 'jmp/images/1-103.png'."""
    st = settings or get_settings()
    # keep only the filename; the corpus has a single flat image namespace
    return st.images_dir / Path(src).name


async def _fetch_image(
    client: httpx.AsyncClient, src: str, st: Settings, sem: asyncio.Semaphore
) -> ImageResult:
    dest = image_path_for(src, st)
    if dest.exists() and dest.stat().st_size > 0:
        return ImageResult(src=src, ok=True, from_cache=True, n_bytes=dest.stat().st_size)

    # srcs are stored decoded (a few contain spaces); re-encode for the request
    url = st.source.page_url(quote(src, safe="/"))
    async with sem:
        for attempt in range(st.crawl.retries):
            try:
                if st.crawl.delay:
                    await asyncio.sleep(random.uniform(0, st.crawl.delay))
                resp = await client.get(url)
                if resp.status_code == 200:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(resp.content)
                    return ImageResult(
                        src=src, ok=True, from_cache=False, n_bytes=len(resp.content)
                    )
                if resp.status_code == 404:
                    return ImageResult(src=src, ok=False, from_cache=False, error="404")
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt == st.crawl.retries - 1:
                    return ImageResult(src=src, ok=False, from_cache=False, error=last)
            await asyncio.sleep((2**attempt) * 0.4 + random.uniform(0, 0.2))

    return ImageResult(src=src, ok=False, from_cache=False, error="exhausted retries")


async def download_images_async(
    srcs: Sequence[str],
    settings: Settings | None = None,
    progress: Callable[[int, int, ImageResult], None] | None = None,
) -> list[ImageResult]:
    st = settings or get_settings()
    st.ensure_dirs()

    sem = asyncio.Semaphore(st.crawl.concurrency)
    limits = httpx.Limits(
        max_connections=st.crawl.concurrency * 2,
        max_keepalive_connections=st.crawl.concurrency,
    )

    results: list[ImageResult] = []
    async with httpx.AsyncClient(
        timeout=st.crawl.timeout,
        headers={"User-Agent": st.crawl.user_agent},
        limits=limits,
        follow_redirects=True,
    ) as client:
        tasks = [asyncio.create_task(_fetch_image(client, s, st, sem)) for s in srcs]
        total = len(tasks)
        for done, coro in enumerate(asyncio.as_completed(tasks), start=1):
            res = await coro
            results.append(res)
            if progress is not None:
                progress(done, total, res)
    return results


def download_images(
    srcs: Iterable[str],
    settings: Settings | None = None,
    progress: Callable[[int, int, ImageResult], None] | None = None,
) -> list[ImageResult]:
    return asyncio.run(download_images_async(list(srcs), settings, progress))
