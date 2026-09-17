"""Configuration for the JMP documentation RAG.

Values come from config.yaml at the repo root, overridden by environment
variables prefixed JMPDOCS_ (nested keys use a double underscore, e.g.
JMPDOCS_LLM__FAST_MODEL=qwen3:4b).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "config.yaml"


class SourceCfg(BaseModel):
    version: str = "19.1"
    base_url: str = "https://www.jmp.com/support/help/en/19.1"
    toc_file: str = "jmp.html"
    book_root: str = "jmp"

    @property
    def toc_url(self) -> str:
        return f"{self.base_url}/{self.toc_file}"

    def page_url(self, rel_path: str) -> str:
        """Absolute URL for a corpus-relative page path such as 'jmp/tabulate.shtml'."""
        return f"{self.base_url}/{rel_path.lstrip('/')}"


class CrawlCfg(BaseModel):
    concurrency: int = 8
    timeout: float = 30.0
    retries: int = 3
    delay: float = 0.05
    user_agent: str = "jmpdocs-local-rag/0.1 (personal documentation assistant)"


class PathsCfg(BaseModel):
    data_dir: Path = Path("./data")


class IndexCfg(BaseModel):
    embed_model: str = "BAAI/bge-base-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    embed_batch_size: int = 64
    chunk_size: int = 1200
    chunk_overlap: int = 200
    min_page_chars: int = 200


class RetrievalCfg(BaseModel):
    k_per_query: int = 20
    k_final: int = 6
    rerank: bool = False
    rerank_candidates: int = 24
    rerank_max_chars: int = 800
    weight_dense: float = 0.6
    weight_bm25: float = 0.4
    page_type_boost: float = 0.15


class LLMCfg(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    answer_model: str = "qwen3:8b"
    fast_model: str = "qwen3:4b"
    temperature: float = 0.1
    num_ctx: int = 8192
    max_context_chars: int = 5000
    num_predict: int = 420
    use_llm_expansion: bool = False
    reasoning: bool | None = False
    max_retrieval_retries: int = 1


class Settings(BaseSettings):
    source: SourceCfg = Field(default_factory=SourceCfg)
    crawl: CrawlCfg = Field(default_factory=CrawlCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    index: IndexCfg = Field(default_factory=IndexCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)

    model_config = SettingsConfigDict(
        env_prefix="JMPDOCS_",
        env_nested_delimiter="__",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        yaml_file=CONFIG_FILE,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: explicit init args > env vars > .env > config.yaml > defaults
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    # ---- derived paths -------------------------------------------------

    @property
    def data_dir(self) -> Path:
        d = self.paths.data_dir
        return d if d.is_absolute() else (REPO_ROOT / d).resolve()

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def knowledge_dir(self) -> Path:
        return self.data_dir / "knowledge"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus.jsonl"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.images_dir, self.knowledge_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
