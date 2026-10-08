"""End-to-end orchestration and API tests: real stage code where possible,
with external services and ffmpeg replaced by fakes."""

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
ALL_STAGES = [StageName.VALIDATE, StageName.NORMALIZE, StageName.TRANSCRIBE,
              StageName.REFINE, StageName.DOCUMENT, StageName.EXPORT]


def fake_validate(src, max_bytes=None):
    return {"format": "wav", "codec": "pcm_s16le", "sample_rate": 16000, "channels": 1, "duration": 10.0}


def fake_normalize(src, out_dir, probe):
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
            "action_items": [
                {"task": "Update the API docs", "owner": "Priya", "deadline": "Friday",
                 "evidence_segment_ids": [1], "evidence_quote": "Priya will update the API docs by Friday."},
                {"task": "Update the API docs", "owner": "Rahul", "deadline": None,
                 "evidence_segment_ids": [1], "evidence_quote": "Priya will update the API docs"},
            ],
        }), "complete")


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(audio, "validate_audio", fake_validate)
    monkeypatch.setattr(audio, "normalize_audio", fake_normalize)
    monkeypatch.setattr(stt, "transcribe", fake_transcribe)
    monkeypatch.setattr(refine.GeminiRefineProvider, "from_settings", classmethod(lambda cls, s: FakeRefiner()))
    monkeypatch.setattr(document.GeminiDocumentProvider, "from_settings",
                        classmethod(lambda cls, s: FakeDocumenter()))


def start_job(tmp_path, store=None):
    store = store or JobStore()
    job = store.create("meeting.wav")
    work = tmp_path / job.id
    work.mkdir(parents=True)
    src = work / "input.wav"
    src.write_bytes(b"audio")
    return store, job, src


def run(store, job, src):
    runner.run_job(job.id, src, store, Settings(_env_file=None))
    return store.get(job.id)


def test_full_pipeline_runs_all_stages_in_order(tmp_path, pipeline):
    store, job, src = start_job(tmp_path)
    job = run(store, job, src)

    assert job.status == JobStatus.DONE and job.error is None and job.error_code is None
    assert [s.name for s in job.stages] == ALL_STAGES
    assert [s.status for s in job.stages] == [StageStatus.DONE] * 6
    assert all(s.message for s in job.stages)
    assert job.current_stage is None
    assert job.raw_transcript.segments[0].text == TEXTS[0], "raw transcript is unchanged"
    assert job.refined_transcript.segments[0].text == "We agreed to use Kubernetes for the cluster."
    assert len(job.record.decisions) == 1 and [a.owner for a in job.record.action_items] == ["Priya"]
    assert len(job.record.rejected_items) == 1, "invented owner withheld"
    assert "1 withheld" in job.stage(StageName.DOCUMENT).message
    assert not src.parent.exists(), "uploaded audio, normalised audio and the job folder are removed"


def test_failure_stops_pipeline_and_keeps_earlier_results(tmp_path, pipeline, monkeypatch):
    def broken_refine(raw, settings=None):
        raise refine.RefineError("The Gemini free-tier rate limit or quota was reached.", code="rate_limited")

    monkeypatch.setattr(refine, "refine", broken_refine)
    store, job, src = start_job(tmp_path)
    job = run(store, job, src)

    assert job.status == JobStatus.FAILED and "rate limit" in job.error
    assert job.error_code == "rate_limited" and job.current_stage is None
    statuses = {s.name: s.status for s in job.stages}
    assert statuses[StageName.TRANSCRIBE] == StageStatus.DONE
    assert statuses[StageName.REFINE] == StageStatus.FAILED
    assert statuses[StageName.DOCUMENT] == statuses[StageName.EXPORT] == StageStatus.SKIPPED
    assert job.raw_transcript is not None and job.record is None
    assert not src.exists()


def test_audio_rejection_is_reported(tmp_path, pipeline, monkeypatch):
    def reject(src, max_bytes=None):
        raise audio.AudioValidationError("The uploaded file is empty (0 bytes).", code="empty")

    monkeypatch.setattr(audio, "validate_audio", reject)
    store, job, src = start_job(tmp_path)
    job = run(store, job, src)
    assert job.stage(StageName.VALIDATE).status == StageStatus.FAILED
    assert job.stage(StageName.NORMALIZE).status == StageStatus.SKIPPED
    assert (job.error, job.error_code) == ("The uploaded file is empty (0 bytes).", "empty")


def test_silent_audio_fails_at_normalisation(tmp_path, pipeline, monkeypatch):
    def silent(src, out_dir, probe):
        raise audio.AudioValidationError("The recording appears to be silent.", code="silent")

    monkeypatch.setattr(audio, "normalize_audio", silent)
    store, job, src = start_job(tmp_path)
    job = run(store, job, src)
    assert job.stage(StageName.VALIDATE).status == StageStatus.DONE
    assert job.stage(StageName.NORMALIZE).status == StageStatus.FAILED
    assert job.error_code == "silent"


def test_ffmpeg_problem_is_reported_without_details(tmp_path, pipeline, monkeypatch):
    def broken(src, out_dir, probe):
        raise audio.AudioToolError("ffmpeg timed out after 1800s")

    monkeypatch.setattr(audio, "normalize_audio", broken)
    store, job, src = start_job(tmp_path)
    job = run(store, job, src)
    assert job.error_code == "audio_tool_error" and "1800" not in job.error


@pytest.mark.parametrize(
    "stage,error",
    [
        ("transcribe", SttError("No speech was detected in the recording.", code="no_speech")),
        ("transcribe", SttError("Speech-to-text is not configured: GROQ_API_KEY is missing.",
                                code="missing_api_key")),
        ("refine", refine.RefineError("Transcript refinement is not configured: GEMINI_API_KEY is missing.",
                                      code="missing_api_key")),
        ("document", document.DocumentError("The documentation model returned output that could not be "
                                            "understood.", code="invalid_response")),
    ],
)
def test_stage_errors_surface_their_message_and_code(tmp_path, pipeline, monkeypatch, stage, error):
    module = {"transcribe": stt, "refine": refine, "document": document}[stage]

    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(module, stage, boom)
    store, job, src = start_job(tmp_path)
    job = run(store, job, src)
    assert job.stage(StageName(stage)).status == StageStatus.FAILED
    assert (job.error, job.error_code) == (error.message, error.code)


def test_unexpected_error_is_generic(tmp_path, pipeline, monkeypatch):
    monkeypatch.setattr(stt, "transcribe", lambda p, settings=None: 1 / 0)
    store, job, src = start_job(tmp_path)
    job = run(store, job, src)
    assert "unexpected internal error" in job.error and "ZeroDivision" not in job.error
    assert job.error_code == "internal_error"


def test_running_job_reports_current_stage(tmp_path, pipeline, monkeypatch):
    seen = {}
    store, job, src = start_job(tmp_path)

    def spy_refine(raw, settings=None):
        current = store.get(job.id)
        seen.update(stage=current.current_stage, status=current.status)
        raise refine.RefineError("x", code="timeout")

    monkeypatch.setattr(refine, "refine", spy_refine)
    run(store, job, src)
    assert seen == {"stage": StageName.REFINE, "status": JobStatus.RUNNING}


def test_job_store_evicts_oldest_finished_jobs():
    store = JobStore(max_jobs=2)
    first = store.create("a.wav")
    first.status = JobStatus.DONE
    store.save(first)
    second = store.create("b.wav")  # still queued: never evicted
    third = store.create("c.wav")
    assert store.get(first.id) is None
    assert store.get(second.id) and store.get(third.id)


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
    assert job["filename"] == "meeting.mp3" and job["status"] == "queued" and job["current_stage"] is None
    assert [s["name"] for s in job["stages"]] == [s.value for s in ALL_STAGES]
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


def test_exports_endpoints_match_the_job(api, tmp_path, pipeline):
    client, _ = api
    store, job, src = start_job(tmp_path / "w", main.store)
    run(store, job, src)

    listing = client.get(f"/jobs/{job.id}/exports").json()
    assert [f["key"] for f in listing] == [
        "raw_transcript", "refined_transcript", "minutes", "decisions", "decisions_json",
        "action_items", "action_items_json", "meeting_record_md", "meeting_record_json",
        *[f"{doc}_{fmt}" for doc in ("raw_transcript", "refined_transcript", "minutes", "decisions",
                                     "action_items", "meeting_record") for fmt in ("pdf", "docx")]]
    pdf = client.get(f"/jobs/{job.id}/exports/action_items_pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF-")
    assert pdf.headers["content-type"] == "application/pdf"
    assert 'filename="action_items.pdf"' in pdf.headers["content-disposition"]
    docx = client.get(f"/jobs/{job.id}/exports/decisions_docx")
    assert docx.status_code == 200 and docx.content.startswith(b"PK"), "DOCX is a zip container"
    assert client.get(f"/jobs/{job.id}/exports").json() == listing, "finished job exports are stable (cached)"
    resp = client.get(f"/jobs/{job.id}/exports/meeting_record_json")
    assert resp.status_code == 200
    assert 'filename="meeting_record.json"' in resp.headers["content-disposition"]
    data, api_job = resp.json(), client.get(f"/jobs/{job.id}").json()
    assert data["meeting_record"] == api_job["record"], "download matches what the UI shows"
    assert data["raw_transcript"] == api_job["raw_transcript"]
    assert data["refined_transcript"] == api_job["refined_transcript"]
    csv_text = client.get(f"/jobs/{job.id}/exports/action_items").text
    assert "Priya" in csv_text and "Rahul" not in csv_text, "withheld items are not exported as tasks"
    assert client.get(f"/jobs/{job.id}/exports/nope").status_code == 404
    assert client.get("/jobs/missing/exports").status_code == 404


def test_health_never_exposes_secret_values(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(main, "get_settings", lambda: Settings(_env_file=None, gemini_api_key="AIzaSECRET123",
                                                               groq_api_key="gsk_SECRET456"))
    body = client.get("/health").text
    assert "SECRET" not in body
