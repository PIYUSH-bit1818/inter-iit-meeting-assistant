import pytest
from pydantic import ValidationError

from app.schemas import (
    PIPELINE_ORDER,
    ActionItem,
    EditDecision,
    Evidence,
    Job,
    JobStatus,
    MeetingRecord,
    RawTranscript,
    Segment,
    StageName,
    StageStatus,
    TranscriptEdit,
)

EV = Evidence(segment_ids=[0], quote="we agreed")


def seg(i: int, text: str = "hello", start: float = 0.0, end: float = 1.0) -> Segment:
    return Segment(id=i, start=start, end=end, text=text)


class TestSegment:
    def test_valid(self):
        assert seg(0).text == "hello"

    def test_end_before_start_rejected(self):
        with pytest.raises(ValidationError):
            Segment(id=0, start=2.0, end=1.0, text="x")

    def test_negative_id_rejected(self):
        with pytest.raises(ValidationError):
            Segment(id=-1, start=0, end=1, text="x")


class TestRawTranscript:
    def test_text_joins_non_empty_segments(self):
        t = RawTranscript(segments=[seg(0, " a "), seg(1, ""), seg(2, "b")], stt_model="m")
        assert t.text == "a b"

    def test_duplicate_segment_ids_rejected(self):
        with pytest.raises(ValidationError):
            RawTranscript(segments=[seg(0), seg(0)], stt_model="m")

    def test_empty_transcript_allowed(self):
        assert RawTranscript(segments=[], stt_model="m").text == ""


class TestEditDecision:
    edit = TranscriptEdit(segment_id=0, original="cube nettees", corrected="Kubernetes")

    def test_rejected_needs_reason(self):
        with pytest.raises(ValidationError):
            EditDecision(edit=self.edit, accepted=False)

    def test_accepted_must_not_have_reason(self):
        with pytest.raises(ValidationError):
            EditDecision(edit=self.edit, accepted=True, rejection_reason="x")

    def test_empty_original_rejected(self):
        with pytest.raises(ValidationError):
            TranscriptEdit(segment_id=0, original="", corrected="x")


class TestActionItem:
    @pytest.mark.parametrize("value", [None, "", "  ", "Unspecified", "unknown", "N/A", "TBD"])
    def test_missing_owner_and_deadline_become_none(self, value):
        item = ActionItem(task="Send report", owner=value, deadline=value, evidence=EV)
        assert item.owner is None and item.deadline is None
        assert item.owner_display == "Unspecified"
        assert item.deadline_display == "Unspecified"

    def test_stated_values_kept(self):
        item = ActionItem(task="Send report", owner=" Priya ", deadline="Friday", evidence=EV)
        assert item.owner == "Priya" and item.deadline == "Friday"

    def test_evidence_required(self):
        with pytest.raises(ValidationError):
            ActionItem(task="Send report")

    def test_evidence_needs_segment_ids(self):
        with pytest.raises(ValidationError):
            Evidence(segment_ids=[], quote="x")


class TestMeetingRecord:
    def test_lists_default_empty(self):
        rec = MeetingRecord(summary="s", document_model="m")
        assert rec.decisions == [] and rec.action_items == []
        assert rec.proposals_not_agreed == [] and rec.minutes == []

    def test_json_roundtrip(self):
        rec = MeetingRecord(
            summary="s",
            action_items=[ActionItem(task="t", evidence=EV)],
            document_model="m",
        )
        again = MeetingRecord.model_validate_json(rec.model_dump_json())
        assert again == rec
        assert again.action_items[0].owner is None


class TestJob:
    def test_new_job_has_all_stages_pending_in_order(self):
        job = Job(id="abc", filename="a.wav")
        assert job.status == JobStatus.QUEUED
        assert [s.name for s in job.stages] == list(PIPELINE_ORDER)
        assert all(s.status == StageStatus.PENDING for s in job.stages)

    def test_stage_lookup(self):
        job = Job(id="abc", filename="a.wav")
        assert job.stage(StageName.REFINE).name == StageName.REFINE
