"""Data contracts passed between pipeline stages.

Flow:
    upload -> RawTranscript (STT) -> RefinedTranscript (LLM A) -> MeetingRecord (LLM B)

Segment ids are stable across stages so every decision/action item can point
back to the exact part of the transcript it came from.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UNSPECIFIED = "Unspecified"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Stage 0: audio validation / normalisation
# ---------------------------------------------------------------------------


class PreparedAudio(BaseModel):
    """A validated upload converted to the format the STT stage expects."""

    source_path: Path
    normalized_path: Path
    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    source_format: str = Field(description="Container reported by ffprobe, e.g. 'mp3'")
    source_codec: str | None = None
    source_sample_rate: int | None = None
    source_channels: int | None = None


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
    stt_provider: str | None = None
    chunk_count: int = Field(default=1, ge=1, description="STT requests used")

    _unique = field_validator("segments")(_check_unique_ids)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


# ---------------------------------------------------------------------------
# Stage 2: domain-aware refinement (LLM A)
# ---------------------------------------------------------------------------


class SpanChange(BaseModel):
    """One edit inside a segment, as reported by the refinement model."""

    original_span: str = Field(min_length=1)
    refined_span: str
    reason: str = ""


class RefinementStatus(str, Enum):
    UNCHANGED = "unchanged"  # model kept the raw text
    REFINED = "refined"  # model edit passed every guard
    FALLBACK = "fallback"  # model edit rejected; raw text kept


class SegmentRefinement(BaseModel):
    """Raw vs refined text for one segment, with the audit trail."""

    segment_id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    original_text: str
    refined_text: str
    changed: bool
    status: RefinementStatus
    changes: list[SpanChange] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list, description="Why a refinement was rejected")
    rejected_text: str | None = Field(default=None, description="The model output that was rejected")

    @model_validator(mode="after")
    def _consistent(self) -> SegmentRefinement:
        same = self.refined_text == self.original_text
        if self.status == RefinementStatus.REFINED:
            if same or not self.changed:
                raise ValueError("a refined segment must differ from the original")
        elif not same or self.changed:
            raise ValueError(f"{self.status.value} segments must keep the original text")
        if self.status == RefinementStatus.FALLBACK and not self.issues:
            raise ValueError("fallback segments must list the issues found")
        if self.status != RefinementStatus.FALLBACK and (self.issues or self.rejected_text):
            raise ValueError("only fallback segments carry issues or rejected text")
        return self


class RefinedTranscript(BaseModel):
    """Separate from RawTranscript, which is never modified.

    ``segments`` mirrors the raw segments one-to-one (same ids, timestamps and
    order) with refined text; ``refinements`` holds the per-segment audit trail.
    """

    segments: list[Segment]
    refinements: list[SegmentRefinement] = Field(default_factory=list)
    refine_model: str
    prompt_version: str | None = None

    _unique = field_validator("segments")(_check_unique_ids)

    @model_validator(mode="after")
    def _refinements_match_segments(self) -> RefinedTranscript:
        if self.refinements and [r.segment_id for r in self.refinements] != [s.id for s in self.segments]:
            raise ValueError("refinements must match segments one-to-one and in order")
        return self

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


# ---------------------------------------------------------------------------
# Stage 3: meeting documentation (LLM B)
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MeetingMinute(_StrictModel):
    """One discussion topic. Minutes describe discussion, not decisions."""

    topic: str = Field(min_length=1)
    discussion: str = Field(min_length=1)
    segment_ids: list[int] = Field(min_length=1)


class Decision(_StrictModel):
    """Something explicitly agreed, confirmed or approved in the meeting."""

    decision: str = Field(min_length=1)
    evidence_segment_ids: list[int] = Field(min_length=1)
    evidence_quote: str = Field(min_length=1, description="Copied exactly from the refined transcript")


class Proposal(_StrictModel):
    """Raised or suggested but not agreed - kept separate from decisions."""

    proposal: str = Field(min_length=1)
    evidence_segment_ids: list[int] = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)


class ActionItem(_StrictModel):
    """An explicitly assigned or committed task.

    ``owner`` and ``deadline`` are None unless the transcript states them.
    """

    task: str = Field(min_length=1)
    owner: str | None = None
    deadline: str | None = None
    evidence_segment_ids: list[int] = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)

    @field_validator("owner", "deadline", mode="before")
    @classmethod
    def _placeholder_is_none(cls, v: object) -> object:
        # Blank or placeholder values mean "not stated": store None, never a guess.
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() in {"unspecified", "unknown", "n/a", "na", "none", "null", "tbd"}:
                return None
            return s
        return v

    @property
    def owner_display(self) -> str:
        return self.owner or UNSPECIFIED

    @property
    def deadline_display(self) -> str:
        return self.deadline or UNSPECIFIED


class RecordItemKind(str, Enum):
    SUMMARY = "summary"
    MINUTE = "minute"
    DECISION = "decision"
    PROPOSAL = "proposal"
    ACTION_ITEM = "action_item"


class RejectedItem(_StrictModel):
    """Model output that failed validation, kept for review (never shown as fact)."""

    kind: RecordItemKind
    content: dict
    reasons: list[str] = Field(min_length=1)


class MeetingRecord(_StrictModel):
    """Structured meeting documentation generated from the refined transcript."""

    summary: str | None = Field(description="None if the generated summary failed validation")
    minutes: list[MeetingMinute] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    proposals: list[Proposal] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    rejected_items: list[RejectedItem] = Field(default_factory=list)
    document_model: str
    prompt_version: str | None = None


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
