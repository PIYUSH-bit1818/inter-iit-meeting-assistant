"""Runs the full pipeline for one job, in order, updating its stage status.

validate+normalise -> transcribe (Groq Whisper) -> refine (Gemini, LLM #1)
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
    work_dir = Path(source).parent
    job.status = JobStatus.RUNNING
    store.save(job)

    def start(name: StageName) -> None:
        stage = job.stage(name)
        stage.status, stage.started_at = StageStatus.RUNNING, _now()
        store.save(job)

    def finish(name: StageName, message: str | None = None) -> None:
        stage = job.stage(name)
        stage.status, stage.finished_at, stage.message = StageStatus.DONE, _now(), message
        store.save(job)

    current = StageName.VALIDATE
    try:
        start(current)
        prepared = audio.prepare_audio(source, work_dir,
                                       max_bytes=settings.max_upload_mb * 1024 * 1024)
        finish(current, f"{prepared.duration_seconds:.1f}s of audio, converted to 16 kHz mono")

        current = StageName.TRANSCRIBE
        start(current)
        job.raw_transcript = stt.transcribe(prepared, settings=settings)
        finish(current, f"{len(job.raw_transcript.segments)} segments ({job.raw_transcript.stt_model})")

        current = StageName.REFINE
        start(current)
        job.refined_transcript = refine.refine(job.raw_transcript, settings=settings)
        changed = sum(r.changed for r in job.refined_transcript.refinements)
        finish(current, f"{changed} segments refined ({job.refined_transcript.refine_model})")

        current = StageName.DOCUMENT
        start(current)
        job.record = document.document(job.refined_transcript, settings=settings)
        r = job.record
        finish(current, f"{len(r.decisions)} decisions, {len(r.proposals)} proposals, "
                        f"{len(r.action_items)} action items ({r.document_model})")

        current = StageName.EXPORT
        start(current)
        files = export.build_exports(job.raw_transcript, job.refined_transcript, job.record)
        finish(current, f"{len(files)} files ready")

        job.status = JobStatus.DONE
    except _USER_ERRORS as exc:
        _fail(job, current, exc.message)
    except AudioToolError as exc:
        log.error("audio tooling failed for job %s: %s", job_id, exc)
        _fail(job, current, "Audio processing is unavailable on the server (FFmpeg problem).")
    except Exception:
        log.exception("unexpected failure in stage %s of job %s", current.value, job_id)
        _fail(job, current, "An unexpected internal error occurred. Please try again.")
    finally:
        # Audio is not kept after processing: only the transcripts and record are.
        (work_dir / audio.NORMALIZED_FILENAME).unlink(missing_ok=True)
        Path(source).unlink(missing_ok=True)
        store.save(job)


def _fail(job, stage_name: StageName, message: str) -> None:
    stage = job.stage(stage_name)
    stage.status, stage.finished_at, stage.message = StageStatus.FAILED, _now(), message
    for s in job.stages:
        if s.status == StageStatus.PENDING:
            s.status = StageStatus.SKIPPED
    job.status = JobStatus.FAILED
    job.error = message


def _now() -> datetime:
    return datetime.now(timezone.utc)
