"""Upload validation and normalisation.

All ffmpeg/ffprobe usage lives in this module. ``prepare_audio`` is the single
entry point: it checks the upload, probes it, converts it to 16 kHz mono
16-bit PCM WAV for the STT stage, and rejects files with no usable audio.

Two error types:
- ``AudioValidationError``: the user's file is the problem. ``message`` is safe
  to show in the UI; ``code`` is a stable identifier for the API/tests.
- ``AudioToolError``: the server is the problem (ffmpeg missing, timeout).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import uuid
import wave
from pathlib import Path

from ..schemas import PreparedAudio

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
)
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
NORMALIZED_FILENAME = "normalized.wav"

MIN_DURATION_SECONDS = 1.0
# Peak level below which a recording is treated as silent. Digital silence
# reports about -91 dB; quiet but real speech peaks well above -40 dB.
SILENCE_THRESHOLD_DB = -50.0

PROBE_TIMEOUT_SECONDS = 60
NORMALIZE_TIMEOUT_SECONDS = 30 * 60

_MAX_VOLUME_RE = re.compile(r"max_volume:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dB")


class AudioValidationError(Exception):
    """The uploaded file cannot be processed. ``message`` is user-facing."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class AudioToolError(RuntimeError):
    """ffmpeg/ffprobe is unavailable or failed for reasons unrelated to the file."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_audio(
    path: Path | str,
    out_dir: Path | str,
    *,
    max_bytes: int | None = None,
    min_duration: float = MIN_DURATION_SECONDS,
    silence_threshold_db: float = SILENCE_THRESHOLD_DB,
) -> PreparedAudio:
    """Validate ``path`` and write ``out_dir/normalized.wav``.

    Equivalent to ``validate_audio`` followed by ``normalize_audio``. On any
    failure no normalized file is left behind in ``out_dir``.
    """
    probe = validate_audio(path, max_bytes=max_bytes, min_duration=min_duration)
    return normalize_audio(path, out_dir, probe, min_duration=min_duration,
                           silence_threshold_db=silence_threshold_db)


def validate_audio(
    path: Path | str,
    *,
    max_bytes: int | None = None,
    min_duration: float = MIN_DURATION_SECONDS,
) -> dict:
    """File checks plus ffprobe. Returns the probe info for ``normalize_audio``."""
    src = Path(path)
    check_file(src, max_bytes=max_bytes)
    probe = probe_audio(src)
    if probe["duration"] is not None and probe["duration"] < min_duration:
        raise _too_short(probe["duration"], min_duration)
    return probe


def normalize_audio(
    path: Path | str,
    out_dir: Path | str,
    probe: dict,
    *,
    min_duration: float = MIN_DURATION_SECONDS,
    silence_threshold_db: float = SILENCE_THRESHOLD_DB,
) -> PreparedAudio:
    """Convert a validated file to 16 kHz mono WAV and reject unusable audio
    (too short after decoding, or silent)."""
    src = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / NORMALIZED_FILENAME
    if out.resolve() == src.resolve():
        raise ValueError("output path would overwrite the source file")

    try:
        max_volume_db = _normalize(src, out)
        duration, rate, channels = _read_wav_params(out)
        if duration < min_duration:
            raise _too_short(duration, min_duration)
        if max_volume_db is None or max_volume_db < silence_threshold_db:
            raise AudioValidationError(
                "The recording appears to be silent - no usable audio was found.",
                code="silent",
            )
    except BaseException:
        out.unlink(missing_ok=True)
        raise

    return PreparedAudio(
        source_path=src,
        normalized_path=out,
        duration_seconds=duration,
        sample_rate=rate,
        channels=channels,
        source_format=probe["format"],
        source_codec=probe["codec"],
        source_sample_rate=probe["sample_rate"],
        source_channels=probe["channels"],
    )


def check_file(path: Path, *, max_bytes: int | None = None) -> None:
    """Cheap checks that need no external tools."""
    if not path.exists():
        raise AudioValidationError("The uploaded file could not be found.", code="not_found")
    if not path.is_file():
        raise AudioValidationError("The upload is not a regular file.", code="not_a_file")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        shown = f"'{ext}'" if ext else "with no extension"
        raise AudioValidationError(
            f"Unsupported file type {shown}. Please upload one of: {_supported_list()}.",
            code="unsupported_format",
        )

    size = path.stat().st_size
    if size == 0:
        raise AudioValidationError("The uploaded file is empty (0 bytes).", code="empty")
    if max_bytes is not None and size > max_bytes:
        raise AudioValidationError(
            f"The file is too large ({size / 1_048_576:.1f} MB). "
            f"Maximum allowed size is {max_bytes / 1_048_576:.0f} MB.",
            code="too_large",
        )


def probe_audio(path: Path) -> dict:
    """Run ffprobe and return info about the first audio stream.

    Keys: format, codec, sample_rate, channels, duration (any may be None
    except format).
    """
    cmd = [
        _tool("ffprobe"),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = _run(cmd, PROBE_TIMEOUT_SECONDS)
    if result.returncode != 0:
        log.info("ffprobe rejected %s: %s", path.name, _tail(result.stderr))
        raise _unreadable()

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        raise _unreadable() from None

    streams = data.get("streams") or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise AudioValidationError(
            "The file does not contain an audio track.", code="no_audio_stream"
        )

    fmt = data.get("format") or {}
    duration = _to_float(fmt.get("duration")) or _to_float(audio.get("duration"))
    return {
        "format": fmt.get("format_name") or path.suffix.lstrip(".").lower(),
        "codec": audio.get("codec_name"),
        "sample_rate": _to_int(audio.get("sample_rate")),
        "channels": _to_int(audio.get("channels")),
        "duration": duration,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _normalize(src: Path, out: Path) -> float | None:
    """Convert the first audio stream to 16 kHz mono PCM WAV.

    Writes to a temporary sibling then renames, so ``out`` only ever holds a
    complete file. Returns the peak level in dB measured during the same pass.
    """
    tmp = out.with_name(f".{out.stem}.{uuid.uuid4().hex}.part")
    cmd = [
        _tool("ffmpeg"),
        "-nostdin", "-hide_banner", "-nostats",
        "-v", "info",  # volumedetect reports at info level
        "-y",
        "-i", str(src),
        "-map", "0:a:0",
        "-vn", "-sn", "-dn",
        "-af", "volumedetect",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(tmp),
    ]
    try:
        result = _run(cmd, NORMALIZE_TIMEOUT_SECONDS)
        if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            log.info("ffmpeg failed on %s: %s", src.name, _tail(result.stderr))
            raise AudioValidationError(
                "The audio could not be decoded. The file may be corrupted.",
                code="decode_failed",
            )
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)

    return parse_max_volume(result.stderr)


def _read_wav_params(path: Path) -> tuple[float, int, int]:
    try:
        with wave.open(str(path), "rb") as w:
            frames, rate, channels = w.getnframes(), w.getframerate(), w.getnchannels()
    except (wave.Error, EOFError) as exc:
        raise AudioValidationError(
            "The audio could not be decoded. The file may be corrupted.",
            code="decode_failed",
        ) from exc
    if rate != TARGET_SAMPLE_RATE or channels != TARGET_CHANNELS:
        raise AudioToolError(
            f"normalisation produced {rate} Hz / {channels} ch instead of "
            f"{TARGET_SAMPLE_RATE} Hz / {TARGET_CHANNELS} ch"
        )
    return (frames / rate if rate else 0.0), rate, channels


def parse_max_volume(ffmpeg_log: str) -> float | None:
    """Extract the last ``max_volume`` reading from ffmpeg volumedetect output."""
    matches = _MAX_VOLUME_RE.findall(ffmpeg_log or "")
    if not matches:
        return None
    return float(matches[-1])  # float() also handles "-inf"


def _tool(name: str) -> str:
    override = os.environ.get(f"{name.upper()}_BINARY")
    path = override or shutil.which(name)
    if not path:
        raise AudioToolError(
            f"{name} is not installed or not on PATH. Install FFmpeg to process audio."
        )
    return path


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioToolError(f"{Path(cmd[0]).stem} timed out after {timeout}s") from exc
    except OSError as exc:
        raise AudioToolError(f"could not run {Path(cmd[0]).stem}: {exc}") from exc


def _unreadable() -> AudioValidationError:
    return AudioValidationError(
        "The file could not be read as audio. It may be corrupted or not a real "
        f"audio file. Supported formats: {_supported_list()}.",
        code="unreadable",
    )


def _too_short(duration: float, minimum: float) -> AudioValidationError:
    return AudioValidationError(
        f"The recording is too short ({duration:.2f}s). "
        f"It must be at least {minimum:g} second(s) long.",
        code="too_short",
    )


def _supported_list() -> str:
    return ", ".join(e.lstrip(".").upper() for e in sorted(SUPPORTED_EXTENSIONS))


def _tail(text: str, lines: int = 5) -> str:
    return " | ".join((text or "").strip().splitlines()[-lines:])


def _to_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_int(v: object) -> int | None:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
