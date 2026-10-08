"""Runs the full pipeline for one job, in order, updating its stage status.

validate -> normalise -> transcribe (Groq Whisper) -> refine (Gemini, LLM #1)
-> document (Gemini, LLM #2) -> export

A failing stage marks the job failed with a user-facing message; later stages
are skipped and results from earlier stages are kept.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings, get_settings
from ..jobs import JobStore
from ..schemas import JobStatus, StageName, StageStatus
from . import audio, document, export, refine, stt
from .audio import AudioToolError, AudioValidationError
from .gemini import LLMError
from .stt import SttError

log = logging.getLogger(__name__)

_USER_ERRORS = (AudioValidationError, SttError, LLMError)


def run_job(job_id: str, source: Path, store: JobStore, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    job = store.get(job_id)
    if job is None:
        raise KeyError(job_id)
    source = Path(source)
    work_dir = source.parent
    job.status = JobStatus.RUNNING
    store.save(job)

    def start(name: StageName) -> None:
        stage = job.stage(name)
        stage.status, stage.started_at = StageStatus.RUNNING, _now()
        job.current_stage = name
        store.save(job)

    def finish(name: StageName, message: str | None = None) -> None:
        stage = job.stage(name)
        stage.status, stage.finished_at, stage.message = StageStatus.DONE, _now(), message
        store.save(job)

    current = StageName.VALIDATE
    try:
        start(current)
        probe = audio.validate_audio(source, max_bytes=settings.max_upload_mb * 1024 * 1024)
        duration = f"{probe['duration']:.1f}s, " if probe["duration"] else ""
        finish(current, f"{duration}{probe['format']} / {probe['codec']}")

        current = StageName.NORMALIZE
        start(current)
        prepared = audio.normalize_audio(source, work_dir, probe)
        finish(current, f"{prepared.duration_seconds:.1f}s converted to 16 kHz mono WAV")

        current = StageName.TRANSCRIBE
        start(current)
        job.raw_transcript = stt.transcribe(prepared, settings=settings)
        raw = job.raw_transcript
        finish(current, f"{len(raw.segments)} segments, {raw.chunk_count} request(s) ({raw.stt_model})")

        current = StageName.REFINE
        start(current)
        job.refined_transcript = refine.refine(job.raw_transcript, settings=settings)
        refs = job.refined_transcript.refinements
        changed = sum(r.changed for r in refs)
        rejected = sum(r.status.value == "fallback" for r in refs)
        finish(current, f"{changed} segments refined, {rejected} edits rejected "
                        f"({job.refined_transcript.refine_model})")

        current = StageName.DOCUMENT
        start(current)
        job.record = document.document(job.refined_transcript, settings=settings)
        r = job.record
        finish(current, f"{len(r.decisions)} decisions, {len(r.proposals)} proposals, "
                        f"{len(r.action_items)} action items, {len(r.rejected_items)} withheld "
                        f"({r.document_model})")

        current = StageName.EXPORT
        start(current)
        files = export.build_exports(job.raw_transcript, job.refined_transcript, job.record,
                                     source_name=job.filename)
        finish(current, f"{len(files)} files ready")

        job.status = JobStatus.DONE
    except _USER_ERRORS as exc:
        _fail(job, current, exc.message, exc.code)
    except AudioToolError as exc:
        log.error("audio tooling failed for job %s: %s", job_id, exc)
        _fail(job, current, "Audio processing is unavailable on the server (FFmpeg problem).",
              "audio_tool_error")
    except Exception:
        log.exception("unexpected failure in stage %s of job %s", current.value, job_id)
        _fail(job, current, "An unexpected internal error occurred. Please try again.", "internal_error")
    finally:
        job.current_stage = None
        # Audio is not kept after processing: only the transcripts and record are.
        (work_dir / audio.NORMALIZED_FILENAME).unlink(missing_ok=True)
        source.unlink(missing_ok=True)
        try:
            work_dir.rmdir()  # only succeeds if nothing else is in it
        except OSError:
            pass
        store.save(job)


def _fail(job, stage_name: StageName, message: str, code: str) -> None:
    stage = job.stage(stage_name)
    stage.status, stage.finished_at, stage.message = StageStatus.FAILED, _now(), message
    for s in job.stages:
        if s.status == StageStatus.PENDING:
            s.status = StageStatus.SKIPPED
    job.status = JobStatus.FAILED
    job.error = message
    job.error_code = code


def _now() -> datetime:
    return datetime.now(timezone.utc)
