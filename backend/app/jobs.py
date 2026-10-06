"""In-memory job store (jobs live only as long as the backend process).

Only the most recent ``max_jobs`` finished jobs are kept, so a long-running
server does not accumulate results indefinitely.
"""

from __future__ import annotations

import threading
import uuid

from .schemas import Job, JobStatus

MAX_JOBS = 50


class JobStore:
    def __init__(self, max_jobs: int = MAX_JOBS) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.max_jobs = max_jobs

    def create(self, filename: str) -> Job:
        job = Job(id=uuid.uuid4().hex, filename=filename)
        with self._lock:
            self._jobs[job.id] = job
            self._evict()
        return job.model_copy(deep=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def save(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job.model_copy(deep=True)

    def _evict(self) -> None:
        # Drop the oldest finished jobs first; never drop a running job.
        excess = len(self._jobs) - self.max_jobs
        if excess <= 0:
            return
        finished = [j for j in self._jobs.values() if j.status in (JobStatus.DONE, JobStatus.FAILED)]
        for job in sorted(finished, key=lambda j: j.created_at)[:excess]:
            del self._jobs[job.id]


store = JobStore()
