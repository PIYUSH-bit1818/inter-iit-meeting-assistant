"""Application settings loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Credentials (SecretStr keeps them out of reprs and logs)
    groq_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # Model IDs - no defaults on purpose; they must be set explicitly
    stt_model: str | None = None
    refine_model: str | None = None
    document_model: str | None = None

    # App
    data_dir: Path = REPO_ROOT / "data"
    max_upload_mb: int = Field(default=100, gt=0)
    allowed_extensions: tuple[str, ...] = (
        ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".mpeg", ".mpga",
    )

    def readiness(self) -> dict[str, bool]:
        """Which pieces of configuration are present (never exposes values)."""
        return {
            "groq_api_key": bool(self.groq_api_key and self.groq_api_key.get_secret_value()),
            "anthropic_api_key": bool(
                self.anthropic_api_key and self.anthropic_api_key.get_secret_value()
            ),
            "stt_model": bool(self.stt_model),
            "refine_model": bool(self.refine_model),
            "document_model": bool(self.document_model),
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
