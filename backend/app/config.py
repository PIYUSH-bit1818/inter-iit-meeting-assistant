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
    gemini_api_key: SecretStr | None = None

    # Speech-to-text (Groq). whisper-large-v3 is Groq's most accurate
    # Whisper model per its docs; override if Groq's model list changes.
    stt_model: str = "whisper-large-v3"
    stt_language: str = "en"
    # Groq upload limit per request: 25 MB on the free tier, 100 MB on dev tier
    groq_max_request_mb: float = Field(default=25, gt=0)
    stt_request_timeout_seconds: float = Field(default=300, gt=0)

    # Transcript refinement (Google Gemini). gemini-3.6-flash was selected from
    # the live Gemini models list: a stable Flash model on the free tier that
    # passed a real structured-JSON check (see TECHNICAL.md).
    refine_model: str = "gemini-3.6-flash"
    # Comma-separated models tried in order if the primary is overloaded (503/429)
    refine_fallback_models: str = ""
    # Gemini 3 thinking level: minimal | low | medium | high
    refine_thinking_level: str = "low"
    refine_timeout_seconds: float = Field(default=300, gt=0)
    # Attempts per model for transient errors (408/429/5xx), with backoff
    refine_retry_attempts: int = Field(default=4, ge=1)
    # Segments are sent in batches of roughly this many words
    refine_max_words_per_request: int = Field(default=1500, gt=0)
    refine_max_output_tokens: int = Field(default=32768, gt=0)

    # Meeting documentation (Google Gemini) - a different model from refinement.
    # gemini-3-flash-preview was the strongest free-tier model that passed a
    # real structured-JSON check (Pro models need billing; see TECHNICAL.md).
    document_model: str = "gemini-3-flash-preview"
    # Tried in order if the primary is overloaded (503/429); verified free tier
    document_fallback_models: str = "gemini-3.5-flash-lite"
    document_thinking_level: str = "medium"
    document_timeout_seconds: float = Field(default=300, gt=0)
    document_retry_attempts: int = Field(default=4, ge=1)
    document_max_output_tokens: int = Field(default=32768, gt=0)

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
            "gemini_api_key": bool(self.gemini_api_key and self.gemini_api_key.get_secret_value()),
            "stt_model": bool(self.stt_model),
            "refine_model": bool(self.refine_model),
            "document_model": bool(self.document_model),
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
