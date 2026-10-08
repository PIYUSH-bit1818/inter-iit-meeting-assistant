import csv
import io
import json

from app.pipeline.export import (
    action_items_csv,
    build_exports,
    complete_record,
    decisions_markdown,
    minutes_markdown,
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
    text = transcript_text(raw.segments, "Raw transcript")
    assert text.startswith("Raw transcript\n")
    assert "[00:00:00 - 00:00:04] (#0) we agreed to use cube nettees" in text
    assert "Kubernetes" in transcript_text(refined.segments)


def test_markdown_contains_every_section_and_not_specified():
    _, refined, record = make()
    md = record_markdown(record, refined.segments)
    for heading in ["## Summary", "## Minutes", "## Decisions", "## Proposals", "## Action items",
                    "## Withheld by validation"]:
        assert heading in md
    assert "| Update the docs | Priya | Not specified |" in md
    assert "| Send \\| report | Not specified | Not specified |" in md  # pipes escaped in table cells
    assert '#0 @ 00:00:00): "We agreed to use Kubernetes."' in md  # evidence with timestamp
    assert "#1 @ 00:00:04" in md


def test_markdown_and_json_carry_the_same_decisions_and_tasks():
    raw, refined, record = make()
    md = record_markdown(record, refined.segments)
    data = complete_record(raw, refined, record)
    for d in data["meeting_record"]["decisions"]:
        assert d["decision"] in md
    for a in data["meeting_record"]["action_items"]:
        assert a["task"].split(" |")[0] in md
    assert data["meeting_record"]["action_items"][1]["owner"] is None  # null in JSON
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
    assert rows[1][:3] == ["Update the docs", "Priya", "Not specified"]


def test_build_exports():
    raw, refined, record = make()
    files = {f.key: f for f in build_exports(raw, refined, record)}
    assert list(files) == ["raw_transcript", "refined_transcript", "minutes", "decisions", "decisions_json",
                           "action_items", "action_items_json", "meeting_record_md", "meeting_record_json",
                           *[f"{doc}_{fmt}" for doc in ("raw_transcript", "refined_transcript", "minutes",
                                                        "decisions", "action_items", "meeting_record")
                             for fmt in ("pdf", "docx")]]
    assert files["action_items_pdf"].filename == "action_items.pdf"
    assert files["action_items_pdf"].mime == "application/pdf"
    assert files["decisions_docx"].filename == "key_decisions.docx"
    assert files["decisions_docx"].mime.endswith("wordprocessingml.document")
    assert b"cube nettees" in files["raw_transcript"].data
    assert b"Kubernetes" in files["refined_transcript"].data
    assert json.loads(files["decisions_json"].data)[0]["decision"] == "Use Kubernetes"
    assert b"Use Kubernetes" in files["decisions"].data and b"Try serverless" in files["decisions"].data
    assert b"## Minutes" in files["minutes"].data and b"## Decisions" not in files["minutes"].data
    assert json.loads(files["action_items_json"].data)[1]["owner"] is None
    assert files["meeting_record_md"].mime == "text/markdown"


def test_full_json_contains_everything_needed_to_reconstruct():
    raw, refined, record = make()
    data = json.loads({f.key: f for f in build_exports(raw, refined, record)}["meeting_record_json"].data)
    assert RawTranscript.model_validate(data["raw_transcript"]) == raw
    assert RefinedTranscript.model_validate(data["refined_transcript"]) == refined
    rebuilt = MeetingRecord.model_validate(data["meeting_record"])
    assert rebuilt == record and rebuilt.rejected_items[0].reasons == ["owner invented"]


def test_partial_exports_after_a_failed_stage():
    raw, _, _ = make()
    files = {f.key: f for f in build_exports(raw, None, None)}
    assert list(files) == ["raw_transcript", "meeting_record_json", "raw_transcript_pdf", "raw_transcript_docx"]
    assert json.loads(files["meeting_record_json"].data)["meeting_record"] is None
    assert build_exports(None, None, None) == []


def test_minutes_and_decisions_markdown():
    _, refined, record = make()
    assert "Kubernetes was chosen." in minutes_markdown(record, refined.segments)
    md = decisions_markdown(record, refined.segments)
    assert "Use Kubernetes" in md and "## Proposals" in md
