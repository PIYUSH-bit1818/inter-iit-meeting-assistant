"""Upload validation and normalisation (ffprobe / ffmpeg). Implemented in Phase 2."""

from pathlib import Path


class AudioValidationError(Exception):
    """Raised with a user-facing message when an upload cannot be processed."""


def validate(path: Path) -> float:
    """Check the file is non-empty, readable audio with speech-length duration.

    Returns the duration in seconds.
    """
    raise NotImplementedError


def normalize(path: Path, out_dir: Path) -> Path:
    """Convert to 16 kHz mono WAV for the STT stage."""
    raise NotImplementedError
