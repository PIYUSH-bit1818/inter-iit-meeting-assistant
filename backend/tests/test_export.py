import csv
import io
import json

from app.pipeline.export import (
    action_items_csv,
    build_exports,
    complete_record,
    record_markdown,
    timestamp,
    transcript_text,
)
from app.schemas import (
    ActionItem,
    Decision,
    MeetingMinute,
    MeetingRecord,
    Proposal,
    RawTranscript,
    RefinedTranscript,
    RejectedItem,
    Segment,
)


def make():
    raw = RawTranscript(segments=[Segment(id=0, start=0, end=4.2, text="we agreed to use cube nettees"),
                                  Segment(id=1, start=4.2, end=3725.5, text="priya will update the docs")],
                        stt_model="whisper-large-v3")
    refined = RefinedTranscript(segments=[Segment(id=0, start=0, end=4.2, text="We agreed to use Kubernetes."),
                                          Segment(id=1, start=4.2, end=3725.5, text="Priya will update the docs.")],
                                refine_model="gemini-3.6-flash")
    record = MeetingRecord(
        summary="The team agreed to use Kubernetes.",
        minutes=[MeetingMinute(topic="Platform", discussion="Kubernetes was chosen.", segment_ids=[0])],
        decisions=[Decision(decision="Use Kubernetes", evidence_segment_ids=[0],
                            evidence_quote="We agreed to use Kubernetes.")],
        proposals=[Proposal(proposal="Try serverless", evidence_segment_ids=[0], evidence_quote="use")],
        action_items=[
            ActionItem(task="Update the docs", owner="Priya", evidence_segment_ids=[1],
                       evidence_quote="Priya will update the docs."),
            ActionItem(task="Send | report", evidence_segment_ids=[1], evidence_quote="will update"),
        ],
        rejected_items=[RejectedItem(kind="action_item", content={"task": "x"}, reasons=["owner invented"])],
        document_model="gemini-3-flash-preview",
        prompt_version="meeting_documentation_v1",
    )
    return raw, refined, record


def test_timestamp_and_transcript_text():
    raw, refined, _ = make()
    assert timestamp(3725.5) == "01:02:05"
    text = transcript_text(raw.segments)
    assert "[00:00:00 - 00:00:04] (#0) we agreed to use cube nettees" in text
    assert "Kubernetes" in transcript_text(refined.segments)


def test_markdown_contains_every_section_and_unspecified():
    _, _, record = make()
    md = record_markdown(record)
    for heading in ["## Summary", "## Minutes", "## Decisions", "## Proposals", "## Action items",
                    "## Withheld by validation"]:
        assert heading in md
    assert "| Update the docs | Priya | Unspecified |" in md
    assert "| Send \\| report | Unspecified | Unspecified |" in md  # pipes escaped in table cells
    assert "Unspecified | Unspecified" in md
    assert '"We agreed to use Kubernetes."' in md


def test_markdown_and_json_carry_the_same_decisions_and_tasks():
    raw, refined, record = make()
    md = record_markdown(record)
    data = complete_record(raw, refined, record)
    for d in data["meeting_record"]["decisions"]:
        assert d["decision"] in md
    for a in data["meeting_record"]["action_items"]:
        assert a["task"].split(" |")[0] in md
    assert data["meeting_record"]["action_items"][1]["owner"] is None  # null in JSON, not "Unspecified"
    assert data["models"] == {"speech_to_text": "whisper-large-v3", "transcript_refinement": "gemini-3.6-flash",
                              "meeting_documentation": "gemini-3-flash-preview"}


def test_withheld_summary_is_marked():
    _, _, record = make()
    record.summary = None
    assert "Summary withheld" in record_markdown(record)


def test_empty_sections_are_explicit():
    md = record_markdown(MeetingRecord(summary="s", document_model="m"))
    assert "No decisions were recorded" in md and "No action items were assigned" in md


def test_action_items_csv():
    _, _, record = make()
    rows = list(csv.reader(io.StringIO(action_items_csv(record))))
    assert rows[0] == ["task", "owner", "deadline", "evidence_segment_ids", "evidence_quote"]
    assert rows[1][:3] == ["Update the docs", "Priya", "Unspecified"]


def test_build_exports():
    raw, refined, record = make()
    files = {f.key: f for f in build_exports(raw, refined, record)}
    assert set(files) == {"raw_transcript", "refined_transcript", "meeting_record_md", "decisions",
                          "action_items", "action_items_csv", "meeting_record_json"}
    assert b"cube nettees" in files["raw_transcript"].data
    assert b"Kubernetes" in files["refined_transcript"].data
    assert json.loads(files["decisions"].data)[0]["decision"] == "Use Kubernetes"
    full = json.loads(files["meeting_record_json"].data)
    assert full["raw_transcript"]["segments"][0]["text"] == "we agreed to use cube nettees"
    assert files["meeting_record_md"].mime == "text/markdown"
