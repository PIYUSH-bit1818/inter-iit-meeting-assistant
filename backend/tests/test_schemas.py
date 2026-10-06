import pytest
from pydantic import ValidationError

from app.schemas import (
    PIPELINE_ORDER,
    ActionItem,
    Decision,
    Job,
    JobStatus,
    MeetingMinute,
    MeetingRecord,
    RawTranscript,
    RefinedTranscript,
    RefinementStatus,
    RejectedItem,
    Segment,
    SegmentRefinement,
    SpanChange,
    StageName,
    StageStatus,
)

EV = {"evidence_segment_ids": [0], "evidence_quote": "we agreed"}


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


def refinement(**kw) -> SegmentRefinement:
    base = dict(segment_id=0, start=0.0, end=1.0, original_text="cube nettees",
                refined_text="cube nettees", changed=False, status=RefinementStatus.UNCHANGED)
    return SegmentRefinement(**{**base, **kw})


class TestSegmentRefinement:
    def test_unchanged(self):
        assert refinement().status == RefinementStatus.UNCHANGED

    def test_refined_must_differ(self):
        with pytest.raises(ValidationError):
            refinement(status=RefinementStatus.REFINED, changed=True)
        r = refinement(status=RefinementStatus.REFINED, changed=True, refined_text="Kubernetes",
                       changes=[SpanChange(original_span="cube nettees", refined_span="Kubernetes")])
        assert r.changed

    def test_fallback_keeps_original_and_needs_issues(self):
        with pytest.raises(ValidationError):
            refinement(status=RefinementStatus.FALLBACK)
        with pytest.raises(ValidationError):
            refinement(status=RefinementStatus.FALLBACK, refined_text="x", issues=["bad"])
        r = refinement(status=RefinementStatus.FALLBACK, issues=["numbers changed"], rejected_text="x")
        assert r.refined_text == r.original_text

    def test_issues_only_on_fallback(self):
        with pytest.raises(ValidationError):
            refinement(issues=["x"])

    def test_empty_original_span_rejected(self):
        with pytest.raises(ValidationError):
            SpanChange(original_span="", refined_span="x")


class TestRefinedTranscript:
    def test_refinements_must_align_with_segments(self):
        with pytest.raises(ValidationError):
            RefinedTranscript(segments=[seg(0), seg(1)], refinements=[refinement(segment_id=1)],
                              refine_model="m")
        ok = RefinedTranscript(segments=[seg(0)], refinements=[refinement()], refine_model="m")
        assert ok.text == "hello"


class TestActionItem:
    @pytest.mark.parametrize("value", [None, "", "  ", "Unspecified", "unknown", "N/A", "TBD"])
    def test_missing_owner_and_deadline_become_none(self, value):
        item = ActionItem(task="Send report", owner=value, deadline=value, **EV)
        assert item.owner is None and item.deadline is None
        assert item.owner_display == "Not specified"
        assert item.deadline_display == "Not specified"

    def test_stated_values_kept(self):
        item = ActionItem(task="Send report", owner=" Priya ", deadline="Friday", **EV)
        assert item.owner == "Priya" and item.deadline == "Friday"

    def test_evidence_required(self):
        with pytest.raises(ValidationError):
            ActionItem(task="Send report")

    def test_evidence_needs_segment_ids_and_quote(self):
        with pytest.raises(ValidationError):
            ActionItem(task="t", evidence_segment_ids=[], evidence_quote="x")
        with pytest.raises(ValidationError):
            ActionItem(task="t", evidence_segment_ids=[0], evidence_quote="")

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ActionItem(task="t", priority="high", **EV)
        with pytest.raises(ValidationError):
            Decision(decision="d", confidence=0.9, **EV)

    def test_minute_needs_segment_ids(self):
        with pytest.raises(ValidationError):
            MeetingMinute(topic="t", discussion="d", segment_ids=[])


class TestMeetingRecord:
    def test_lists_default_empty(self):
        rec = MeetingRecord(summary="s", document_model="m")
        assert rec.decisions == [] and rec.action_items == []
        assert rec.proposals == [] and rec.minutes == [] and rec.rejected_items == []

    def test_summary_may_be_withheld(self):
        assert MeetingRecord(summary=None, document_model="m").summary is None

    def test_rejected_item_needs_reasons(self):
        with pytest.raises(ValidationError):
            RejectedItem(kind="decision", content={}, reasons=[])

    def test_json_roundtrip(self):
        rec = MeetingRecord(
            summary="s",
            action_items=[ActionItem(task="t", **EV)],
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
