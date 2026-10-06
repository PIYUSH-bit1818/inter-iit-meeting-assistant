"""FastAPI entry point. Run from backend/: uvicorn app.main:app --reload"""

import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, UploadFile

from . import __version__
from .config import get_settings
from .jobs import store
from .pipeline.audio import SUPPORTED_EXTENSIONS
from .pipeline.export import build_exports
from .pipeline.runner import run_job
from .schemas import Job

app = FastAPI(title="Meeting Assistant API", version=__version__)


@app.get("/health")
def health() -> dict:
    """Liveness plus a config readiness report (booleans only, no secret values)."""
    return {
        "status": "ok",
        "version": __version__,
        "config": get_settings().readiness(),
    }


@app.post("/jobs", status_code=202)
def create_job(file: UploadFile) -> Job:
    """Upload a recording and start processing it in the background."""
    settings = get_settings()
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(e.lstrip(".").upper() for e in sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(415, f"Unsupported file type '{ext or 'none'}'. Please upload one of: {supported}.")

    jobs_dir = settings.data_dir / "jobs"
    incoming = jobs_dir / f"_incoming_{uuid.uuid4().hex}"
    incoming.mkdir(parents=True, exist_ok=True)
    upload = incoming / f"input{ext}"
    limit = settings.max_upload_mb * 1024 * 1024
    with upload.open("wb") as out:
        copied = 0
        while chunk := file.file.read(1024 * 1024):
            copied += len(chunk)
            if copied > limit:
                break
            out.write(chunk)
    if copied > limit:
        shutil.rmtree(incoming, ignore_errors=True)
        raise HTTPException(413, f"The file is larger than the {settings.max_upload_mb} MB limit.")

    job = store.create(name)
    work_dir = jobs_dir / job.id
    incoming.rename(work_dir)
    threading.Thread(target=run_job, args=(job.id, work_dir / upload.name, store), daemon=True).start()
    return job


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job


def _exports(job_id: str):
    job = get_job(job_id)
    return {f.key: f for f in build_exports(job.raw_transcript, job.refined_transcript, job.record)}


@app.get("/jobs/{job_id}/exports")
def list_exports(job_id: str) -> list[dict]:
    """Downloadable files for a job, built from the same objects as GET /jobs/{id}."""
    return [{"key": f.key, "label": f.label, "filename": f.filename, "mime": f.mime, "size": len(f.data)}
            for f in _exports(job_id).values()]


@app.get("/jobs/{job_id}/exports/{key}")
def download_export(job_id: str, key: str) -> Response:
    file = _exports(job_id).get(key)
    if file is None:
        raise HTTPException(404, "That export is not available for this job.")
    return Response(content=file.data, media_type=file.mime,
                    headers={"Content-Disposition": f'attachment; filename="{file.filename}"'})
