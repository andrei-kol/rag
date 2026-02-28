"""
Project-wide settings loaded from environment variables.

Usage:
    from src.config import settings

    print(settings.openai_api_key)
    print(settings.qdrant_url)

All values can be overridden via environment variables or a .env file.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # --- LangSmith (optional) ---
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="rag-financial-docs", alias="LANGCHAIN_PROJECT")

    # --- Qdrant ---
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="financial_docs", alias="QDRANT_COLLECTION")

    # --- Embedding ---
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    embedding_batch_size: int = Field(default=100, alias="EMBEDDING_BATCH_SIZE")

    # --- LLM ---
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=10, alias="RETRIEVAL_TOP_K")
    reranker_top_n: int = Field(default=3, alias="RERANKER_TOP_N")

    # --- Chunking ---
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")

    # --- Paths ---
    project_root: Path = Path(__file__).parent.parent
    data_raw_dir: Path = Field(default=Path("data/raw"), alias="DATA_RAW_DIR")
    data_processed_dir: Path = Field(
        default=Path("data/processed"), alias="DATA_PROCESSED_DIR"
    )
    data_eval_dir: Path = Field(default=Path("data/eval"), alias="DATA_EVAL_DIR")

    def resolved(self, path: Path) -> Path:
        """Resolve a relative path against the project root."""
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def raw_dir(self) -> Path:
        return self.resolved(self.data_raw_dir)

    @property
    def processed_dir(self) -> Path:
        return self.resolved(self.data_processed_dir)

    @property
    def eval_dir(self) -> Path:
        return self.resolved(self.data_eval_dir)


# Singleton — import this everywhere
settings = Settings()
