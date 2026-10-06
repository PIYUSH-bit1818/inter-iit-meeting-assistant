"""Stage 1: speech-to-text (Groq Whisper). Implemented in a later phase."""

from pathlib import Path

from ..schemas import RawTranscript


def transcribe(audio_path: Path) -> RawTranscript:
    raise NotImplementedError
