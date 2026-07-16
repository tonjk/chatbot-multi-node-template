"""Environment-backed application configuration."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "chatbot-multi-node-template"
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_request_timeout_seconds: float = 30.0

    auth_username: str = ""
    auth_password_hash: SecretStr = SecretStr("")
    jwt_secret: SecretStr = SecretStr("")

    database_url: str = "sqlite:///./.data/chatbot.db"
    chroma_persist_directory: Path = Path("./.data/chroma")
    knowledge_directory: Path = Path("./knowledge")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    def require_openai(self) -> None:
        if not self.openai_api_key.get_secret_value() or not self.openai_model.strip():
            raise ValueError("OPENAI_API_KEY and OPENAI_MODEL are required")
