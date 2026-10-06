"""End-to-end orchestration tests: real stage code where possible, with the
external services (ffmpeg excluded) replaced by fakes."""

import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.jobs import JobStore
from app.pipeline import audio, document, refine, runner, stt
from app.pipeline.gemini import ProviderReply
from app.pipeline.stt import SttError
from app.schemas import JobStatus, PreparedAudio, RawTranscript, Segment, StageName, StageStatus

TEXTS = ["we agreed to use cube nettees for the cluster", "priya will update the a p i docs by friday"]


def fake_prepare(src, out_dir, max_bytes=None):
    wav = out_dir / audio.NORMALIZED_FILENAME
    wav.write_bytes(b"RIFF")
    return PreparedAudio(source_path=src, normalized_path=wav, duration_seconds=10, sample_rate=16000,
                         channels=1, source_format="wav")


def fake_transcribe(prepared, settings=None):
    return RawTranscript(segments=[Segment(id=i, start=i * 5.0, end=i * 5.0 + 4, text=t) for i, t in enumerate(TEXTS)],
                         stt_model="whisper-large-v3", stt_provider="groq")


class FakeRefiner:
    name, model = "fake", "refine-model"

    def complete(self, system, context, request):
        payload = json.loads(request[request.index("{"):])
        fixed = {TEXTS[0]: "We agreed to use Kubernetes for the cluster.",
                 TEXTS[1]: "Priya will update the API docs by Friday."}
        return ProviderReply(json.dumps({"segments": [
            {"segment_id": s["segment_id"], "original_text": s["text"], "refined_text": fixed[s["text"]],
             "changed": True, "changes": []} for s in payload["segments"]]}), "complete")


class FakeDocumenter:
    name, model = "fake", "doc-model"

    def complete(self, system, request):
        return ProviderReply(json.dumps({
            "summary": "The team agreed to use Kubernetes.",
            "minutes": [{"topic": "Platform", "discussion": "Kubernetes was agreed.", "segment_ids": [0]}],
            "decisions": [{"decision": "Use Kubernetes for the cluster", "evidence_segment_ids": [0],
                           "evidence_quote": "We agreed to use Kubernetes for the cluster."}],
            "proposals": [],
            "action_items": [{"task": "Update the API docs", "owner": "Priya", "deadline": "Friday",
                              "evidence_segment_ids": [1], "evidence_quote": "Priya will update the API docs by Friday."}],
        }), "complete")


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(audio, "prepare_audio", fake_prepare)
    monkeypatch.setattr(stt, "transcribe", fake_transcribe)
    monkeypatch.setattr(refine.GeminiRefineProvider, "from_settings", classmethod(lambda cls, s: FakeRefiner()))
    monkeypatch.setattr(document.GeminiDocumentProvider, "from_settings",
                        classmethod(lambda cls, s: FakeDocumenter()))


def start_job(tmp_path):
    store = JobStore()
    job = store.create("meeting.wav")
    src = tmp_path / "input.wav"
    src.write_bytes(b"audio")
    return store, job, src


def test_full_pipeline_runs_all_stages_in_order(tmp_path, pipeline):
    store, job, src = start_job(tmp_path)
    runner.run_job(job.id, src, store, Settings(_env_file=None))
    job = store.get(job.id)

    assert job.status == JobStatus.DONE and job.error is None
    assert [s.status for s in job.stages] == [StageStatus.DONE] * 5
    assert job.raw_transcript.segments[0].text == TEXTS[0], "raw transcript is unchanged"
    assert job.refined_transcript.segments[0].text == "We agreed to use Kubernetes for the cluster."
    assert len(job.record.decisions) == 1 and job.record.action_items[0].owner == "Priya"
    assert job.record.rejected_items == []
    assert "1 decisions" in job.stage(StageName.DOCUMENT).message
    assert not src.exists() and not (tmp_path / audio.NORMALIZED_FILENAME).exists(), "audio is not kept"


def test_failure_stops_pipeline_and_keeps_earlier_results(tmp_path, pipeline, monkeypatch):
    def broken_refine(raw, settings=None):
        raise refine.RefineError("The Gemini free-tier rate limit or quota was reached.", code="rate_limited")

    monkeypatch.setattr(refine, "refine", broken_refine)
    store, job, src = start_job(tmp_path)
    runner.run_job(job.id, src, store, Settings(_env_file=None))
    job = store.get(job.id)

    assert job.status == JobStatus.FAILED and "rate limit" in job.error
    statuses = {s.name: s.status for s in job.stages}
    assert statuses[StageName.TRANSCRIBE] == StageStatus.DONE
    assert statuses[StageName.REFINE] == StageStatus.FAILED
    assert statuses[StageName.DOCUMENT] == statuses[StageName.EXPORT] == StageStatus.SKIPPED
    assert job.raw_transcript is not None and job.record is None


def test_audio_rejection_is_reported(tmp_path, pipeline, monkeypatch):
    def reject(src, out_dir, max_bytes=None):
        raise audio.AudioValidationError("The uploaded file is empty (0 bytes).", code="empty")

    monkeypatch.setattr(audio, "prepare_audio", reject)
    store, job, src = start_job(tmp_path)
    runner.run_job(job.id, src, store, Settings(_env_file=None))
    job = store.get(job.id)
    assert job.stage(StageName.VALIDATE).status == StageStatus.FAILED
    assert job.error == "The uploaded file is empty (0 bytes)."


def test_stt_error_and_unexpected_error(tmp_path, pipeline, monkeypatch):
    monkeypatch.setattr(stt, "transcribe", lambda p, settings=None: (_ for _ in ()).throw(
        SttError("No speech was detected in the recording.", code="no_speech")))
    store, job, src = start_job(tmp_path)
    runner.run_job(job.id, src, store, Settings(_env_file=None))
    assert store.get(job.id).error == "No speech was detected in the recording."

    monkeypatch.setattr(stt, "transcribe", lambda p, settings=None: 1 / 0)
    store, job, src = start_job(tmp_path)
    runner.run_job(job.id, src, store, Settings(_env_file=None))
    assert "unexpected internal error" in store.get(job.id).error


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture
def api(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(main, "get_settings", lambda: Settings(_env_file=None, data_dir=tmp_path, max_upload_mb=1))
    monkeypatch.setattr(main, "run_job", lambda job_id, source, store: started.append((job_id, source)))
    return TestClient(main.app), started


def test_upload_starts_a_job(api):
    client, started = api
    resp = client.post("/jobs", files={"file": ("meeting.mp3", b"ID3fake", "audio/mpeg")})
    assert resp.status_code == 202
    job = resp.json()
    assert job["filename"] == "meeting.mp3" and job["status"] == "queued"
    [(job_id, source)] = started
    assert job_id == job["id"] and source.name == "input.mp3" and source.read_bytes() == b"ID3fake"
    assert client.get(f"/jobs/{job_id}").json()["id"] == job_id


def test_upload_rejects_unsupported_type(api):
    client, started = api
    resp = client.post("/jobs", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 415 and "Unsupported file type" in resp.json()["detail"]
    assert started == []


def test_upload_rejects_oversized_file(api, tmp_path):
    client, started = api
    resp = client.post("/jobs", files={"file": ("big.wav", b"\0" * (1024 * 1024 + 1), "audio/wav")})
    assert resp.status_code == 413 and started == []
    assert not any((tmp_path / "jobs").iterdir()), "no leftover upload"


def test_unknown_job_is_404(api):
    client, _ = api
    assert client.get("/jobs/nope").status_code == 404
