"""Tests for audio validation/normalisation.

Fixtures are synthesised with ffmpeg into pytest's tmp_path, so no binary
audio is stored in the repo and everything is cleaned up automatically.
"""

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from app.pipeline import audio
from app.pipeline.audio import (
    NORMALIZED_FILENAME,
    AudioToolError,
    AudioValidationError,
    parse_max_volume,
    prepare_audio,
)
from app.schemas import PreparedAudio

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")

# extension -> extra ffmpeg output args
CODECS = {
    ".wav": ["-c:a", "pcm_s16le"],
    ".mp3": ["-c:a", "libmp3lame"],
    ".m4a": ["-c:a", "aac"],
    ".ogg": ["-c:a", "libvorbis"],
    ".flac": ["-c:a", "flac"],
    ".webm": ["-c:a", "libopus", "-ar", "48000"],  # Opus cannot encode at 44.1 kHz
    ".mp4": ["-c:a", "aac"],
}


def make_audio(
    path: Path,
    *,
    duration: float = 2.0,
    rate: int = 44_100,
    channels: int = 2,
    silent: bool = False,
    with_video: bool = False,
) -> Path:
    layout = "stereo" if channels == 2 else "mono"
    src = (
        f"anullsrc=r={rate}:cl={layout}"
        if silent
        else f"sine=frequency=440:sample_rate={rate}:duration={duration}"
    )
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-t", str(duration), "-i", src]
    if with_video:
        cmd += ["-f", "lavfi", "-t", str(duration), "-i", "testsrc=size=64x64:rate=10",
                "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += ["-ac", str(channels), "-ar", str(rate), *CODECS[path.suffix.lower()], str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def make_video_only(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-t", "2",
         "-i", "testsrc=size=64x64:rate=10", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def wav_params(path: Path) -> tuple[int, int, int, float]:
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes() / w.getframerate()


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "out"


def assert_no_output(out_dir: Path) -> None:
    leftovers = list(out_dir.iterdir()) if out_dir.exists() else []
    assert leftovers == [], f"unexpected files left behind: {leftovers}"


# ---------------------------------------------------------------------------
# Valid input
# ---------------------------------------------------------------------------


@needs_ffmpeg
@pytest.mark.parametrize("ext", sorted(CODECS))
def test_valid_audio_each_supported_format(tmp_path, out_dir, ext):
    src = make_audio(tmp_path / f"meeting{ext}", duration=2.0)
    result = prepare_audio(src, out_dir)

    assert isinstance(result, PreparedAudio)
    assert result.normalized_path == out_dir / NORMALIZED_FILENAME
    assert result.normalized_path.is_file()
    assert result.sample_rate == 16_000 and result.channels == 1
    assert result.duration_seconds == pytest.approx(2.0, abs=0.15)
    assert result.source_path == src
    assert src.exists(), "source file must not be modified or removed"


@needs_ffmpeg
def test_mp4_with_video_track_uses_audio(tmp_path, out_dir):
    src = make_audio(tmp_path / "call.mp4", with_video=True)
    result = prepare_audio(src, out_dir)
    assert result.channels == 1 and result.sample_rate == 16_000


@needs_ffmpeg
def test_uppercase_extension_accepted(tmp_path, out_dir):
    src = make_audio(tmp_path / "x.wav")
    upper = src.rename(tmp_path / "MEETING.WAV")
    assert prepare_audio(upper, out_dir).channels == 1


# ---------------------------------------------------------------------------
# Normalisation output
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_normalization_output_is_16bit_pcm_wav(tmp_path, out_dir):
    src = make_audio(tmp_path / "in.mp3", duration=3.0)
    result = prepare_audio(src, out_dir)

    rate, channels, sampwidth, duration = wav_params(result.normalized_path)
    assert (rate, channels, sampwidth) == (16_000, 1, 2)
    assert duration == pytest.approx(result.duration_seconds)
    assert [p.name for p in out_dir.iterdir()] == [NORMALIZED_FILENAME], "no temp files left"


@needs_ffmpeg
def test_stereo_becomes_mono(tmp_path, out_dir):
    src = make_audio(tmp_path / "stereo.wav", channels=2)
    assert wav_params(src)[1] == 2

    result = prepare_audio(src, out_dir)
    assert result.source_channels == 2
    assert result.channels == 1
    assert wav_params(result.normalized_path)[1] == 1


@needs_ffmpeg
@pytest.mark.parametrize("rate", [8_000, 22_050, 44_100, 48_000])
def test_any_sample_rate_becomes_16k(tmp_path, out_dir, rate):
    src = make_audio(tmp_path / f"r{rate}.wav", rate=rate, channels=1)
    result = prepare_audio(src, out_dir)
    assert result.source_sample_rate == rate
    assert result.sample_rate == 16_000
    assert wav_params(result.normalized_path)[0] == 16_000
    assert result.duration_seconds == pytest.approx(2.0, abs=0.05)


@needs_ffmpeg
def test_rerun_overwrites_previous_output(tmp_path, out_dir):
    prepare_audio(make_audio(tmp_path / "a.wav", duration=2.0), out_dir)
    second = prepare_audio(make_audio(tmp_path / "b.wav", duration=4.0), out_dir)
    assert wav_params(second.normalized_path)[3] == pytest.approx(4.0, abs=0.05)


# ---------------------------------------------------------------------------
# Rejections (cheap checks, no ffmpeg needed)
# ---------------------------------------------------------------------------


def test_missing_file(tmp_path, out_dir):
    with pytest.raises(AudioValidationError) as exc:
        prepare_audio(tmp_path / "nope.wav", out_dir)
    assert exc.value.code == "not_found"
    assert_no_output(out_dir)


def test_directory_is_rejected(tmp_path, out_dir):
    d = tmp_path / "folder.wav"
    d.mkdir()
    with pytest.raises(AudioValidationError) as exc:
        prepare_audio(d, out_dir)
    assert exc.value.code == "not_a_file"


def test_empty_file(tmp_path, out_dir):
    src = tmp_path / "empty.mp3"
    src.write_bytes(b"")
    with pytest.raises(AudioValidationError) as exc:
        prepare_audio(src, out_dir)
    assert exc.value.code == "empty"
    assert "empty" in exc.value.message.lower()
    assert_no_output(out_dir)


@pytest.mark.parametrize("name", ["notes.txt", "slides.pdf", "clip.avi", "noext"])
def test_unsupported_extension(tmp_path, out_dir, name):
    src = tmp_path / name
    src.write_bytes(b"some content")
    with pytest.raises(AudioValidationError) as exc:
        prepare_audio(src, out_dir)
    assert exc.value.code == "unsupported_format"
    assert "MP3" in exc.value.message and "WAV" in exc.value.message
    assert_no_output(out_dir)


def test_too_large(tmp_path, out_dir):
    src = tmp_path / "big.wav"
    src.write_bytes(b"\0" * 2048)
    with pytest.raises(AudioValidationError) as exc:
        prepare_audio(src, out_dir, max_bytes=1024)
    assert exc.value.code == "too_large"


# ---------------------------------------------------------------------------
# Rejections that need ffmpeg
# ---------------------------------------------------------------------------


@needs_ffmpeg
@pytest.mark.parametrize(
    "name,payload",
    [
        ("corrupt.mp3", b"this is definitely not audio data\n" * 200),
        ("corrupt.wav", b"RIFF\x00\x00\x00\x00WAVEjunkjunkjunk" * 50),
        ("corrupt.m4a", b"\x00\x01\x02\x03garbage" * 500),
    ],
)
def test_corrupt_file(tmp_path, out_dir, name, payload):
    src = tmp_path / name
    src.write_bytes(payload)
    with pytest.raises(AudioValidationError) as exc:
        prepare_audio(src, out_dir)
    assert exc.value.code in {"unreadable", "decode_failed", "no_audio_stream"}
    assert_no_output(out_dir)


@needs_ffmpeg
def test_truncated_file(tmp_path, out_dir):
    good = make_audio(tmp_path / "good.m4a", duration=3.0)
    bad = tmp_path / "truncated.m4a"
    bad.write_bytes(good.read_bytes()[:200])
    with pytest.raises(AudioValidationError):
        prepare_audio(bad, out_dir)
    assert_no_output(out_dir)


@needs_ffmpeg
def test_video_without_audio(tmp_path, out_dir):
    src = make_video_only(tmp_path / "screen.mp4")
    with pytest.raises(AudioValidationError) as exc:
        prepare_audio(src, out_dir)
    assert exc.value.code == "no_audio_stream"
    assert_no_output(out_dir)


@needs_ffmpeg
def test_very_short_audio(tmp_path, out_dir):
    src = make_audio(tmp_path / "blip.wav", duration=0.3)
    with pytest.raises(AudioValidationError) as exc:
        prepare_audio(src, out_dir)
    assert exc.value.code == "too_short"
    assert "too short" in exc.value.message
    assert_no_output(out_dir)


@needs_ffmpeg
def test_silent_audio(tmp_path, out_dir):
    src = make_audio(tmp_path / "silence.wav", duration=3.0, silent=True)
    with pytest.raises(AudioValidationError) as exc:
        prepare_audio(src, out_dir)
    assert exc.value.code == "silent"
    assert_no_output(out_dir)


# ---------------------------------------------------------------------------
# Tooling / helpers
# ---------------------------------------------------------------------------


def test_missing_ffmpeg_is_a_tool_error(tmp_path, out_dir, monkeypatch):
    src = tmp_path / "a.wav"
    src.write_bytes(b"RIFF")
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    monkeypatch.delenv("FFPROBE_BINARY", raising=False)
    with pytest.raises(AudioToolError):
        prepare_audio(src, out_dir)


@pytest.mark.parametrize(
    "log,expected",
    [
        ("[Parsed_volumedetect_0 @ 0x1] max_volume: -3.2 dB", -3.2),
        ("max_volume: -91.0 dB", -91.0),
        ("max_volume: 0.0 dB", 0.0),
        ("max_volume: -inf dB", float("-inf")),
        ("max_volume: -10.0 dB\nmax_volume: -20.5 dB", -20.5),
        ("no volume info here", None),
        ("", None),
    ],
)
def test_parse_max_volume(log, expected):
    assert parse_max_volume(log) == expected


# ---------------------------------------------------------------------------
# validate_audio / normalize_audio (the two pipeline stages)
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_validate_then_normalize(tmp_path, out_dir):
    src = make_audio(tmp_path / "m.mp3", duration=2.0, rate=44_100, channels=2)
    probe = audio.validate_audio(src)
    assert probe["codec"] == "mp3" and probe["channels"] == 2
    assert not out_dir.exists(), "validation writes nothing"
    result = audio.normalize_audio(src, out_dir, probe)
    assert (result.sample_rate, result.channels) == (16_000, 1)


@needs_ffmpeg
def test_validate_rejects_corrupt_and_short_without_normalising(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio at all\n" * 100)
    with pytest.raises(AudioValidationError):
        audio.validate_audio(bad)
    with pytest.raises(AudioValidationError) as exc:
        audio.validate_audio(make_audio(tmp_path / "s.wav", duration=0.3))
    assert exc.value.code == "too_short"


@needs_ffmpeg
def test_normalize_rejects_silence(tmp_path, out_dir):
    src = make_audio(tmp_path / "q.wav", duration=3.0, silent=True)
    probe = audio.validate_audio(src)  # a silent file is still a valid file
    with pytest.raises(AudioValidationError) as exc:
        audio.normalize_audio(src, out_dir, probe)
    assert exc.value.code == "silent"
    assert_no_output(out_dir)
