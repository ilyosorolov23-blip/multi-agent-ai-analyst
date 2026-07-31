"""
F1 - Shared config. Loads every key from .env once, exposes a single
Settings object the rest of the app imports. Nothing here talks to a
network; it just reads env vars and applies sane defaults so optional
services (Tavily, Langfuse, DeepSeek) can be absent without crashing.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # required — all LLM/embedding calls now go through the LiteLLM proxy
    # instead of hitting Gemini directly.
    proxy_api_key: str = _get("PROXY_API_KEY")
    proxy_base_url: str = _get("PROXY_BASE_URL", "https://saidazam-litellm-proxy.hf.space/v1")
    proxy_chat_model: str = _get("PROXY_CHAT_MODEL", "gemini-flash-lite")
    proxy_embedding_model: str = _get("PROXY_EMBEDDING_MODEL", "gemini-embedding")

    # kept only so old .env files with this var don't crash config loading;
    # no code path reads it anymore.
    google_api_key: str = _get("GOOGLE_API_KEY")

    # vector store
    qdrant_url: str = _get("QDRANT_URL")
    qdrant_api_key: str = _get("QDRANT_API_KEY")
    qdrant_collection: str = _get("QDRANT_COLLECTION", "analyst_docs")
    memory_collection: str = _get("MEMORY_COLLECTION", "analyst_memory")

    # optional services
    tavily_api_key: str = _get("TAVILY_API_KEY")
    langfuse_public_key: str = _get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = _get("LANGFUSE_SECRET_KEY")
    langfuse_host: str = _get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    deepseek_api_key: str = _get("DEEPSEEK_API_KEY")

    # app behaviour
    sqlite_db_path: str = _get("SQLITE_DB_PATH", "./data/company.db")
    code_sandbox_timeout: int = int(_get("CODE_SANDBOX_TIMEOUT_SECONDS", "5"))
    supervisor_step_limit: int = int(_get("SUPERVISOR_STEP_LIMIT", "8"))
    cors_origins: str = _get("CORS_ORIGINS", "http://localhost:3000")
    port: int = int(_get("PORT", "8000"))

    @property
    def has_web(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def has_langfuse(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()

if not settings.proxy_api_key:
    raise RuntimeError(
        "PROXY_API_KEY is missing. Copy backend/.env.example to backend/.env "
        "and set PROXY_API_KEY to your LiteLLM proxy key."
    )
