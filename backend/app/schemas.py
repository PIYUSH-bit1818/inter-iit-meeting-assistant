"""Data contracts passed between pipeline stages.

Flow:
    upload -> RawTranscript (STT) -> RefinedTranscript (LLM A) -> MeetingRecord (LLM B)

Segment ids are stable across stages so every decision/action item can point
back to the exact part of the transcript it came from.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

UNSPECIFIED = "Unspecified"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Stage 1: speech-to-text
# ---------------------------------------------------------------------------


class Segment(BaseModel):
    id: int = Field(ge=0)
    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(ge=0, description="End time in seconds")
    text: str

    @model_validator(mode="after")
    def _end_after_start(self) -> Segment:
        if self.end < self.start:
            raise ValueError("segment end must be >= start")
        return self


def _check_unique_ids(segments: list[Segment]) -> list[Segment]:
    ids = [s.id for s in segments]
    if len(ids) != len(set(ids)):
        raise ValueError("segment ids must be unique")
    return segments


class RawTranscript(BaseModel):
    segments: list[Segment]
    language: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    stt_model: str

    _unique = field_validator("segments")(_check_unique_ids)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


# ---------------------------------------------------------------------------
# Stage 2: domain-aware refinement (LLM A)
# ---------------------------------------------------------------------------


class TranscriptEdit(BaseModel):
    """A single correction proposed by the refinement model."""

    segment_id: int = Field(ge=0)
    original: str = Field(min_length=1, description="Exact span from the raw segment")
    corrected: str
    reason: str = ""


class EditDecision(BaseModel):
    """Outcome of running guards on a proposed edit."""

    edit: TranscriptEdit
    accepted: bool
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def _reason_iff_rejected(self) -> EditDecision:
        if not self.accepted and not self.rejection_reason:
            raise ValueError("rejected edits must carry a rejection_reason")
        if self.accepted and self.rejection_reason:
            raise ValueError("accepted edits must not carry a rejection_reason")
        return self


class RefinedTranscript(BaseModel):
    segments: list[Segment]
    edits: list[EditDecision] = Field(default_factory=list)
    refine_model: str

    _unique = field_validator("segments")(_check_unique_ids)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


# ---------------------------------------------------------------------------
# Stage 3: meeting documentation (LLM B)
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """Where in the refined transcript an item is grounded."""

    segment_ids: list[int] = Field(min_length=1)
    quote: str = Field(min_length=1)


class MinutesSection(BaseModel):
    topic: str = Field(min_length=1)
    points: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    """Something explicitly agreed in the meeting (not merely proposed)."""

    text: str = Field(min_length=1)
    evidence: Evidence


class Proposal(BaseModel):
    """Raised or suggested but not agreed - kept separate from decisions."""

    text: str = Field(min_length=1)
    evidence: Evidence


class ActionItem(BaseModel):
    task: str = Field(min_length=1)
    owner: str | None = None
    deadline: str | None = None
    evidence: Evidence

    @field_validator("owner", "deadline", mode="before")
    @classmethod
    def _blank_is_unspecified(cls, v: object) -> object:
        # Blank or placeholder values mean "not stated" - store as None, never guess.
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() in {"unspecified", "unknown", "n/a", "none", "tbd"}:
                return None
            return s
        return v

    @property
    def owner_display(self) -> str:
        return self.owner or UNSPECIFIED

    @property
    def deadline_display(self) -> str:
        return self.deadline or UNSPECIFIED


class MeetingRecord(BaseModel):
    summary: str
    minutes: list[MinutesSection] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    proposals_not_agreed: list[Proposal] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    document_model: str


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


class StageName(str, Enum):
    VALIDATE = "validate"
    TRANSCRIBE = "transcribe"
    REFINE = "refine"
    DOCUMENT = "document"
    EXPORT = "export"


PIPELINE_ORDER: tuple[StageName, ...] = tuple(StageName)


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"  # only used for stages after a failure


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class StageState(BaseModel):
    name: StageName
    status: StageStatus = StageStatus.PENDING
    message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class Job(BaseModel):
    id: str
    filename: str
    created_at: datetime = Field(default_factory=_utcnow)
    status: JobStatus = JobStatus.QUEUED
    error: str | None = None
    stages: list[StageState] = Field(
        default_factory=lambda: [StageState(name=n) for n in PIPELINE_ORDER]
    )
    raw_transcript: RawTranscript | None = None
    refined_transcript: RefinedTranscript | None = None
    record: MeetingRecord | None = None

    def stage(self, name: StageName) -> StageState:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(name)
