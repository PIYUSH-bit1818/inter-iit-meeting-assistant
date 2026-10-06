"""In-memory job store. The pipeline runner is added in a later phase."""

from __future__ import annotations

import threading
import uuid

from .schemas import Job


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        job = Job(id=uuid.uuid4().hex, filename=filename)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def save(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job.model_copy(deep=True)


store = JobStore()
