"""Stage 1: speech-to-text.

Consumes the 16 kHz mono WAV from ``audio.prepare_audio`` and returns a
``RawTranscript`` of timestamped segments, exactly as the provider returned
them (only outer whitespace is trimmed). Nothing here rewrites, corrects or
fills in transcript text.

Long recordings are split into overlapping windows so each request stays
under the provider's upload limit. Each window "owns" a time range; a segment
is kept only from the window that owns its midpoint, which keeps text at
chunk boundaries from being duplicated or dropped. Timestamps are shifted
back to the original recording's timeline.

Provider-specific code lives in ``GroqWhisperProvider``; another provider only
needs ``name``, ``model``, ``max_request_bytes`` and ``transcribe_wav``.
"""

from __future__ import annotations

import io
import logging
import math
import wave
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import Settings, get_settings
from ..schemas import PreparedAudio, RawTranscript, Segment

log = logging.getLogger(__name__)

# Longest audio window per request. Shorter windows mean faster individual
# requests; the byte limit may force them shorter still.
MAX_WINDOW_SECONDS = 600.0
# Audio shared between neighbouring windows. Must exceed half of a typical
# Whisper segment (they are at most ~30 s, usually far shorter).
OVERLAP_SECONDS = 10.0
# Fraction of the provider byte limit we actually use (multipart overhead).
REQUEST_SIZE_SAFETY = 0.95
_WAV_HEADER_BYTES = 1024  # generous allowance for the header


class SttError(Exception):
    """Transcription failed. ``message`` is user-facing; ``code`` is stable."""

    def __init__(self, message: str, code: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable


class SttProvider(Protocol):
    name: str
    model: str
    max_request_bytes: int

    def transcribe_wav(self, wav_bytes: bytes, filename: str) -> dict[str, Any]:
        """Return a verbose-JSON-like dict: ``{"text": str, "segments": [{start, end, text}]}``
        with timestamps relative to the start of ``wav_bytes``."""
        ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe(
    audio: PreparedAudio,
    *,
    provider: SttProvider | None = None,
    settings: Settings | None = None,
) -> RawTranscript:
    settings = settings or get_settings()
    provider = provider or GroqWhisperProvider.from_settings(settings)

    with wave.open(str(audio.normalized_path), "rb") as wav:
        rate, width, channels = wav.getframerate(), wav.getsampwidth(), wav.getnchannels()
        total_frames = wav.getnframes()
        duration = total_frames / rate
        bytes_per_second = rate * width * channels

        windows = plan_chunks(
            duration,
            max_window_seconds=max_window_seconds(provider.max_request_bytes, bytes_per_second),
        )
        log.info("transcribing %.1fs of audio in %d request(s)", duration, len(windows))

        collected: list[tuple[float, float, str]] = []
        for w in windows:
            wav_bytes = _slice_wav(wav, w.start, w.end)
            response = provider.transcribe_wav(wav_bytes, f"chunk_{w.index:03d}.wav")
            for start, end, text in parse_segments(response):
                abs_start = min(start + w.start, duration)
                abs_end = min(max(end, start) + w.start, duration)
                if w.owns((abs_start + abs_end) / 2):
                    collected.append((abs_start, abs_end, text))

    collected.sort(key=lambda s: (s[0], s[1]))
    segments = [
        Segment(id=i, start=round(s, 3), end=round(e, 3), text=t)
        for i, (s, e, t) in enumerate(collected)
    ]
    if not any(s.text for s in segments):
        raise SttError(
            "No speech was detected in the recording.", code="no_speech"
        )

    return RawTranscript(
        segments=segments,
        language=settings.stt_language,
        duration_seconds=round(duration, 3),
        stt_model=provider.model,
        stt_provider=provider.name,
        chunk_count=len(windows),
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkWindow:
    index: int
    start: float  # audio sent to the provider: [start, end)
    end: float
    own_start: float  # segments whose midpoint falls in [own_start, own_end) are kept
    own_end: float

    def owns(self, t: float) -> bool:
        return self.own_start <= t < self.own_end


def max_window_seconds(max_request_bytes: int, bytes_per_second: int) -> float:
    budget = max_request_bytes * REQUEST_SIZE_SAFETY - _WAV_HEADER_BYTES
    seconds = math.floor(budget / bytes_per_second) if budget > 0 else 0
    if seconds <= 2 * OVERLAP_SECONDS:
        raise SttError(
            "The speech-to-text request size limit is too small to process audio.",
            code="config_error",
        )
    return min(MAX_WINDOW_SECONDS, float(seconds))


def plan_chunks(
    duration: float,
    *,
    max_window_seconds: float,
    overlap: float = OVERLAP_SECONDS,
) -> list[ChunkWindow]:
    """Split ``[0, duration)`` into windows of at most ``max_window_seconds``.

    Each window owns a contiguous ``step``-long range and extends ``overlap``
    seconds either side of it so boundary segments are heard in full.
    """
    if duration <= max_window_seconds:
        return [ChunkWindow(0, 0.0, duration, -math.inf, math.inf)]

    step = max_window_seconds - 2 * overlap
    if step <= 0:
        raise ValueError("max_window_seconds must exceed twice the overlap")

    count = math.ceil(duration / step)
    windows = []
    for i in range(count):
        own_start = i * step
        own_end = min((i + 1) * step, duration)
        windows.append(
            ChunkWindow(
                index=i,
                start=max(0.0, own_start - overlap),
                end=min(duration, own_end + overlap),
                own_start=-math.inf if i == 0 else own_start,
                own_end=math.inf if i == count - 1 else own_end,
            )
        )
    return windows


def _slice_wav(wav: wave.Wave_read, start: float, end: float) -> bytes:
    rate = wav.getframerate()
    first = int(round(start * rate))
    last = min(int(round(end * rate)), wav.getnframes())
    wav.setpos(first)
    frames = wav.readframes(last - first)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(wav.getnchannels())
        out.setsampwidth(wav.getsampwidth())
        out.setframerate(rate)
        out.writeframes(frames)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Response parsing (provider-neutral verbose-JSON shape)
# ---------------------------------------------------------------------------


def parse_segments(response: Any) -> list[tuple[float, float, str]]:
    """Validate a verbose-JSON response and return ``(start, end, text)`` tuples."""
    if not isinstance(response, dict):
        raise SttError(
            "The speech-to-text service returned an unsupported response format.",
            code="unsupported_response",
        )

    segments = response.get("segments")
    if segments is None:
        # Text with no timestamps cannot be mapped to segments without inventing times.
        if str(response.get("text") or "").strip():
            raise SttError(
                "The speech-to-text service returned text without timestamps.",
                code="unsupported_response",
            )
        return []  # no speech in this chunk
    if not isinstance(segments, list):
        raise _malformed("'segments' is not a list")

    parsed = []
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise _malformed(f"segment {i} is not an object")
        start, end, text = seg.get("start"), seg.get("end"), seg.get("text")
        if not _is_time(start) or not _is_time(end):
            raise _malformed(f"segment {i} has invalid timestamps")
        if not isinstance(text, str):
            raise _malformed(f"segment {i} has no text")
        parsed.append((float(start), float(end), text.strip()))
    return parsed


def _is_time(v: Any) -> bool:
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
        and v >= 0
    )


def _malformed(detail: str) -> SttError:
    log.warning("malformed STT response: %s", detail)
    return SttError(
        "The speech-to-text service returned an invalid response.", code="invalid_response"
    )


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------


class GroqWhisperProvider:
    name = "groq"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        language: str = "en",
        max_request_bytes: int,
        timeout: float = 300,
        client: Any = None,
    ) -> None:
        self.model = model
        self.language = language
        self.max_request_bytes = max_request_bytes
        if client is None:
            from groq import Groq

            client = Groq(api_key=api_key, timeout=timeout, max_retries=3)
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings, *, client: Any = None) -> GroqWhisperProvider:
        key = settings.groq_api_key.get_secret_value().strip() if settings.groq_api_key else ""
        if not key:
            raise SttError(
                "Speech-to-text is not configured: GROQ_API_KEY is missing. "
                "Add it to your .env file.",
                code="missing_api_key",
            )
        return cls(
            api_key=key,
            model=settings.stt_model,
            language=settings.stt_language,
            max_request_bytes=int(settings.groq_max_request_mb * 1_000_000),
            timeout=settings.stt_request_timeout_seconds,
            client=client,
        )

    def transcribe_wav(self, wav_bytes: bytes, filename: str) -> dict[str, Any]:
        import groq

        if len(wav_bytes) > self.max_request_bytes:
            raise SttError(
                "An audio chunk exceeded the speech-to-text upload limit.",
                code="request_too_large",
            )
        try:
            result = self._client.audio.transcriptions.create(
                file=(filename, wav_bytes),
                model=self.model,
                language=self.language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                temperature=0.0,
            )
        except groq.AuthenticationError as exc:
            raise SttError(
                "The Groq API key was rejected. Check GROQ_API_KEY.", code="auth_failed"
            ) from exc
        except groq.PermissionDeniedError as exc:
            raise SttError(
                "The Groq API key does not have access to this model.", code="auth_failed"
            ) from exc
        except groq.NotFoundError as exc:
            raise SttError(
                f"Speech-to-text model '{self.model}' was not found. Check STT_MODEL.",
                code="model_not_found",
            ) from exc
        except groq.RateLimitError as exc:
            raise SttError(
                "The speech-to-text service rate limit was reached. Please try again shortly.",
                code="rate_limited",
                retryable=True,
            ) from exc
        except groq.APITimeoutError as exc:
            raise SttError(
                "The speech-to-text service timed out.", code="timeout", retryable=True
            ) from exc
        except groq.APIConnectionError as exc:
            raise SttError(
                "Could not reach the speech-to-text service. Check the network connection.",
                code="network_error",
                retryable=True,
            ) from exc
        except groq.APIStatusError as exc:
            raise _status_error(exc) from exc

        return _to_dict(result)


def _status_error(exc: Any) -> SttError:
    status = getattr(exc, "status_code", None)
    if status == 413:
        return SttError(
            "The audio was too large for the speech-to-text service.",
            code="request_too_large",
        )
    if status is not None and status >= 500:
        return SttError(
            "The speech-to-text service had an internal error. Please try again.",
            code="provider_error",
            retryable=True,
        )
    log.warning("Groq returned HTTP %s: %s", status, exc)
    return SttError(
        f"The speech-to-text service rejected the request (HTTP {status}).",
        code="provider_rejected",
    )


def _to_dict(result: Any) -> dict[str, Any] | Any:
    # The SDK returns a pydantic model whose segments live in extra fields.
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result  # parse_segments reports it as unsupported
