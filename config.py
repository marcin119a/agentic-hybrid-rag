import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter — LLM (chat models)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # OpenAI — embeddings (text-embedding-3-small)
    openai_api_key: str

    # LangSmith tracing (opcjonalne)
    langsmith_api_key: str = ""
    langsmith_tracing: str = "false"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "hybrid-rag-ai-programming"

    # Chroma — flaga przebudowy indeksu
    rebuild_index: bool = False


settings = Settings()

# OpenAI i LangSmith czytają klucze bezpośrednio z os.environ — propagujemy tu z .env
os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
if settings.langsmith_api_key:
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
os.environ.setdefault("LANGSMITH_TRACING", settings.langsmith_tracing)
os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
