"""Tests for the STT stage. No network calls and no real API key: the provider
or the Groq client is replaced with fakes."""

import io
import math
import wave
from pathlib import Path
from types import SimpleNamespace

import groq
import httpx
import pytest
from groq.types.audio import Transcription

from app.config import Settings
from app.pipeline import stt
from app.pipeline.stt import (
    GroqWhisperProvider,
    SttError,
    max_window_seconds,
    parse_segments,
    plan_chunks,
    transcribe,
)
from app.schemas import PreparedAudio

RATE = 16_000


def write_wav(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"\0\0" * int(seconds * RATE))
    return path


def prepared(tmp_path: Path, seconds: float) -> PreparedAudio:
    wav = write_wav(tmp_path / "normalized.wav", seconds)
    return PreparedAudio(
        source_path=tmp_path / "in.mp3", normalized_path=wav, duration_seconds=seconds,
        sample_rate=RATE, channels=1, source_format="mp3",
    )


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / w.getframerate()


def settings(**kw) -> Settings:
    base = {"groq_api_key": "test-key", "_env_file": None}
    return Settings(**{**base, **kw})


class FakeProvider:
    """Returns canned responses, or builds one per call via ``respond(call_index, seconds)``."""

    name = "fake"
    model = "fake-whisper"

    def __init__(self, responses=None, respond=None, max_request_bytes=25_000_000):
        self.responses = list(responses or [])
        self.respond = respond
        self.max_request_bytes = max_request_bytes
        self.calls: list[tuple[str, float, int]] = []

    def transcribe_wav(self, wav_bytes, filename):
        secs = wav_seconds(wav_bytes)
        self.calls.append((filename, secs, len(wav_bytes)))
        if self.respond:
            return self.respond(len(self.calls) - 1, secs)
        return self.responses.pop(0)


def seg(start, end, text):
    return {"id": 99, "seek": 0, "start": start, "end": end, "text": text, "no_speech_prob": 0.01}


# ---------------------------------------------------------------------------
# Successful transcription (single request)
# ---------------------------------------------------------------------------


def test_successful_transcription(tmp_path):
    provider = FakeProvider([{
        "text": " Hello team. We will not ship 3.2 today.",
        "segments": [seg(0.0, 2.5, " Hello team."), seg(2.5, 6.04, " We will not ship 3.2 today.")],
    }])
    result = transcribe(prepared(tmp_path, 10), provider=provider, settings=settings())

    assert [s.text for s in result.segments] == ["Hello team.", "We will not ship 3.2 today."]
    assert result.text == "Hello team. We will not ship 3.2 today."
    assert result.stt_model == "fake-whisper" and result.stt_provider == "fake"
    assert result.chunk_count == 1 and len(provider.calls) == 1
    assert result.duration_seconds == pytest.approx(10)
    assert result.language == "en"


def test_text_is_preserved_verbatim_apart_from_outer_whitespace(tmp_path):
    raw = "  um, so the k8s cluster, uh, isn't ready - 15 nodes, not 50.  "
    provider = FakeProvider([{"text": raw, "segments": [seg(0, 4, raw)]}])
    result = transcribe(prepared(tmp_path, 5), provider=provider, settings=settings())
    assert result.segments[0].text == raw.strip()


def test_segment_ids_are_sequential_and_ignore_provider_ids(tmp_path):
    provider = FakeProvider([{"segments": [seg(0, 1, "a"), seg(1, 2, "b"), seg(2, 3, "c")]}])
    result = transcribe(prepared(tmp_path, 5), provider=provider, settings=settings())
    assert [s.id for s in result.segments] == [0, 1, 2]


def test_timestamps_mapped(tmp_path):
    provider = FakeProvider([{"segments": [seg(0.12, 1.875, "a"), seg(1.875, 4.5, "b")]}])
    result = transcribe(prepared(tmp_path, 5), provider=provider, settings=settings())
    assert [(s.start, s.end) for s in result.segments] == [(0.12, 1.875), (1.875, 4.5)]


def test_segments_sorted_chronologically(tmp_path):
    provider = FakeProvider([{"segments": [seg(4, 5, "c"), seg(0, 1, "a"), seg(2, 3, "b")]}])
    result = transcribe(prepared(tmp_path, 6), provider=provider, settings=settings())
    assert [s.text for s in result.segments] == ["a", "b", "c"]
    assert [s.id for s in result.segments] == [0, 1, 2]


def test_timestamps_clamped_to_recording_length(tmp_path):
    provider = FakeProvider([{"segments": [seg(3.0, 9.9, "runs past the end"), seg(2, 1.5, "end<start")]}])
    result = transcribe(prepared(tmp_path, 4), provider=provider, settings=settings())
    for s in result.segments:
        assert 0 <= s.start <= s.end <= 4


# ---------------------------------------------------------------------------
# Empty / no speech
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        {"text": "", "segments": []},
        {"text": ""},
        {"text": "   ", "segments": [seg(0, 1, "   ")]},
    ],
)
def test_empty_transcript_raises_no_speech(tmp_path, response):
    with pytest.raises(SttError) as exc:
        transcribe(prepared(tmp_path, 3), provider=FakeProvider([response]), settings=settings())
    assert exc.value.code == "no_speech"


# ---------------------------------------------------------------------------
# Malformed / unsupported responses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response,code",
    [
        ("plain text response", "unsupported_response"),
        (None, "unsupported_response"),
        ({"text": "hello but no timestamps"}, "unsupported_response"),
        ({"segments": "not a list"}, "invalid_response"),
        ({"segments": ["not an object"]}, "invalid_response"),
        ({"segments": [{"end": 1, "text": "x"}]}, "invalid_response"),
        ({"segments": [{"start": "0", "end": 1, "text": "x"}]}, "invalid_response"),
        ({"segments": [{"start": -1, "end": 1, "text": "x"}]}, "invalid_response"),
        ({"segments": [{"start": math.nan, "end": 1, "text": "x"}]}, "invalid_response"),
        ({"segments": [{"start": True, "end": 1, "text": "x"}]}, "invalid_response"),
        ({"segments": [{"start": 0, "end": 1}]}, "invalid_response"),
        ({"segments": [{"start": 0, "end": 1, "text": 42}]}, "invalid_response"),
    ],
)
def test_malformed_response(tmp_path, response, code):
    with pytest.raises(SttError) as exc:
        transcribe(prepared(tmp_path, 3), provider=FakeProvider([response]), settings=settings())
    assert exc.value.code == code


def test_parse_segments_accepts_integer_times():
    assert parse_segments({"segments": [{"start": 0, "end": 2, "text": " hi "}]}) == [(0.0, 2.0, "hi")]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_short_audio_is_a_single_window():
    [w] = plan_chunks(300, max_window_seconds=600)
    assert (w.start, w.end) == (0, 300) and w.owns(0) and w.owns(299.9)


def test_plan_chunks_covers_recording_without_gaps():
    windows = plan_chunks(1800, max_window_seconds=600, overlap=10)
    assert windows[0].start == 0 and windows[-1].end == 1800
    for w in windows:
        assert w.end - w.start <= 600
    # ownership ranges tile the whole timeline exactly once
    for t in [0, 0.5, 579.99, 580, 1159.99, 1160, 1799.99]:
        assert sum(w.owns(t) for w in windows) == 1, t
    # neighbouring windows overlap so boundary speech is heard in full
    for a, b in zip(windows, windows[1:]):
        assert a.end - b.start == pytest.approx(20)


def test_max_window_respects_byte_limit():
    bps = RATE * 2
    assert max_window_seconds(25_000_000, bps) == 600  # capped by MAX_WINDOW_SECONDS
    secs = max_window_seconds(2_000_000, bps)
    assert secs * bps + 1024 <= 2_000_000 * 0.95 + 1
    with pytest.raises(SttError) as exc:
        max_window_seconds(100_000, bps)
    assert exc.value.code == "config_error"


def grid_response(call_index, seconds):
    """Pretend speech: one 2-second segment after another across the chunk."""
    n = int(seconds // 2)
    return {"segments": [seg(k * 2.0, k * 2.0 + 2.0, f"c{call_index}-{k}") for k in range(n)]}


def test_chunked_transcription_timestamps_and_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr(stt, "MAX_WINDOW_SECONDS", 60.0)  # force chunking on a short file
    provider = FakeProvider(respond=grid_response)

    result = transcribe(prepared(tmp_path, 130), provider=provider, settings=settings())

    assert result.chunk_count == len(provider.calls) == 4
    assert all(secs <= 60 for _, secs, _ in provider.calls)
    # Every 2 s slot of the original timeline appears exactly once, in order
    assert [s.start for s in result.segments] == [float(t) for t in range(0, 130, 2)]
    assert [s.id for s in result.segments] == list(range(65))
    assert all(a.end <= b.start for a, b in zip(result.segments, result.segments[1:]))
    # later segments come from later chunks (offset applied, not chunk-relative)
    assert result.segments[-1].text.startswith("c3-")
    assert result.segments[-1].end == pytest.approx(130)


def test_chunks_respect_provider_byte_limit(tmp_path):
    limit = 1_000_000  # ~29 s of 16 kHz mono PCM
    provider = FakeProvider(respond=grid_response, max_request_bytes=limit)
    result = transcribe(prepared(tmp_path, 100), provider=provider, settings=settings())
    assert result.chunk_count > 1
    assert all(size <= limit for _, _, size in provider.calls)


def test_chunk_with_no_speech_does_not_fail_whole_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(stt, "MAX_WINDOW_SECONDS", 60.0)

    def respond(i, secs):
        return {"text": "", "segments": []} if i == 1 else grid_response(i, secs)

    result = transcribe(prepared(tmp_path, 130), provider=FakeProvider(respond=respond), settings=settings())
    assert result.segments and result.chunk_count == 4


# ---------------------------------------------------------------------------
# Groq provider: configuration, request shape, error mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_api_key(tmp_path, key):
    with pytest.raises(SttError) as exc:
        transcribe(prepared(tmp_path, 3), settings=settings(groq_api_key=key))
    assert exc.value.code == "missing_api_key"
    assert "GROQ_API_KEY" in exc.value.message


class FakeGroqClient:
    def __init__(self, result=None, error=None):
        self.kwargs = None
        self._result, self._error = result, error
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        if self._error:
            raise self._error
        return self._result


def groq_provider(client, **settings_kw) -> GroqWhisperProvider:
    return GroqWhisperProvider.from_settings(settings(**settings_kw), client=client)


def test_groq_request_and_sdk_response(tmp_path):
    sdk_result = Transcription(
        text=" Budget is 40k.",
        segments=[seg(0.0, 1.6, " Budget is 40k.")],
        language="English",
    )
    client = FakeGroqClient(result=sdk_result)
    result = transcribe(prepared(tmp_path, 3), provider=groq_provider(client), settings=settings())

    assert result.segments[0].text == "Budget is 40k."
    assert result.stt_model == "whisper-large-v3" and result.stt_provider == "groq"
    kw = client.kwargs
    assert kw["model"] == "whisper-large-v3"
    assert kw["response_format"] == "verbose_json"
    assert kw["timestamp_granularities"] == ["segment"]
    assert kw["language"] == "en" and kw["temperature"] == 0.0
    filename, data = kw["file"]
    assert filename.endswith(".wav") and wav_seconds(data) == pytest.approx(3)


def test_groq_model_configurable(tmp_path):
    client = FakeGroqClient(result={"segments": [seg(0, 1, "x")]})
    provider = groq_provider(client, stt_model="whisper-large-v3-turbo")
    transcribe(prepared(tmp_path, 2), provider=provider, settings=settings())
    assert client.kwargs["model"] == "whisper-large-v3-turbo"


REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions")


def status_error(cls, status):
    return cls("boom", response=httpx.Response(status, request=REQ), body=None)


@pytest.mark.parametrize(
    "error,code,retryable",
    [
        (status_error(groq.AuthenticationError, 401), "auth_failed", False),
        (status_error(groq.PermissionDeniedError, 403), "auth_failed", False),
        (status_error(groq.NotFoundError, 404), "model_not_found", False),
        (status_error(groq.RateLimitError, 429), "rate_limited", True),
        (status_error(groq.APIStatusError, 413), "request_too_large", False),
        (status_error(groq.InternalServerError, 500), "provider_error", True),
        (status_error(groq.InternalServerError, 503), "provider_error", True),
        (status_error(groq.BadRequestError, 400), "provider_rejected", False),
        (groq.APITimeoutError(request=REQ), "timeout", True),
        (groq.APIConnectionError(request=REQ), "network_error", True),
    ],
)
def test_groq_api_failures(tmp_path, error, code, retryable):
    provider = groq_provider(FakeGroqClient(error=error))
    with pytest.raises(SttError) as exc:
        transcribe(prepared(tmp_path, 3), provider=provider, settings=settings())
    assert exc.value.code == code
    assert exc.value.retryable is retryable
    assert "test-key" not in exc.value.message


def test_groq_rejects_oversized_chunk_before_sending():
    client = FakeGroqClient(result={"segments": []})
    provider = groq_provider(client, groq_max_request_mb=1)
    with pytest.raises(SttError) as exc:
        provider.transcribe_wav(b"\0" * 1_000_001, "big.wav")
    assert exc.value.code == "request_too_large"
    assert client.kwargs is None, "must not call the API"
