from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jmpdocs.config import get_settings  # noqa: E402
from jmpdocs.retrieval.store import index_exists  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "needs_index: requires the built FAISS/BM25 index"
    )


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def store(settings):
    if not index_exists(settings):
        pytest.skip("index not built; run scripts/build_corpus.py && build_index.py")
    from jmpdocs.retrieval.store import get_store

    return get_store()
