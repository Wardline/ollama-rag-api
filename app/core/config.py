# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # LLM
    llm_provider: str = "ollama"

    # облачная модель
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None

    ollama_host: str = "http://ollama:11434"
    ollama_model: str = "llama3.1"

    # Embeddings
    embedding_backend: str = "sbert"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG
    rag_index_path: str = "/app/app/rag/store" 
    k_top: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
