"""Tests for the meeting-documentation stage (LLM #2). No network and no real
API key: the provider (or the Gemini client) is replaced with fakes."""

import json
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from app.config import Settings
from app.pipeline.document import (
    OUTPUT_SCHEMA,
    PROMPT_PATH,
    PROMPT_VERSION,
    DocumentError,
    GeminiDocumentProvider,
    document,
    load_prompt,
)
from app.pipeline.gemini import ProviderReply
from app.schemas import RecordItemKind, RefinedTranscript, Segment

FAKE_KEY = "AIzaFAKE-doc-0123456789-secret"

# A realistic meeting: decisions (incl. a negative one), a proposal, two
# action items (owner/no owner, deadline/no deadline), numbers, a date, a
# technical term, a negation, and lines that must NOT become decisions/tasks.
MEETING = [
    "Okay, let's start with the infrastructure update.",                                  # 0
    "We agreed to upgrade the Kubernetes cluster to version 1.29 on October 15.",         # 1
    "Maybe we could also move the nightly backups to 2 AM, but that's just an idea.",     # 2
    "Priya will update the API documentation by Friday.",                                 # 3
    "I'll send the cost report to the finance team.",                                     # 4
    "We decided not to change the database schema this sprint.",                          # 5
    "The budget for the upgrade is $42,000, about 15 percent over plan.",                  # 6
    "Someone should review the monitoring alerts.",                                       # 7
    "We need to update the onboarding guide.",                                            # 8
    "Has everyone agreed to the new on-call rotation?",                                   # 9
]


def settings(**kw) -> Settings:
    base = {"gemini_api_key": FAKE_KEY, "document_model": "doc-model", "refine_model": "refine-model",
            "document_fallback_models": "", "_env_file": None}
    return Settings(**{**base, **kw})


def refined_transcript(texts=MEETING) -> RefinedTranscript:
    return RefinedTranscript(
        segments=[Segment(id=i, start=i * 5.0, end=i * 5.0 + 4.0, text=t) for i, t in enumerate(texts)],
        refine_model="refine-model",
    )


def decision(text, ids, quote):
    return {"decision": text, "evidence_segment_ids": ids, "evidence_quote": quote}


def proposal(text, ids, quote):
    return {"proposal": text, "evidence_segment_ids": ids, "evidence_quote": quote}


def action(task, ids, quote, owner=None, deadline=None):
    return {"task": task, "owner": owner, "deadline": deadline, "evidence_segment_ids": ids, "evidence_quote": quote}


def output(summary="The team discussed the infrastructure update.", minutes=None, decisions=(),
           proposals=(), actions=()):
    return {
        "summary": summary,
        "minutes": minutes if minutes is not None else [
            {"topic": "Infrastructure update", "discussion": "The Kubernetes upgrade and its budget were discussed.",
             "segment_ids": [1, 6]}],
        "decisions": list(decisions),
        "proposals": list(proposals),
        "action_items": list(actions),
    }


GOOD = output(
    summary="The team agreed to upgrade the Kubernetes cluster to version 1.29 and not to change the "
            "database schema this sprint. The upgrade budget is $42,000.",
    decisions=[
        decision("Upgrade the Kubernetes cluster to version 1.29 on October 15.", [1],
                 "We agreed to upgrade the Kubernetes cluster to version 1.29 on October 15."),
        decision("Do not change the database schema this sprint.", [5],
                 "We decided not to change the database schema this sprint."),
    ],
    proposals=[proposal("Move the nightly backups to 2 AM.", [2],
                        "Maybe we could also move the nightly backups to 2 AM")],
    actions=[
        action("Update the API documentation", [3], "Priya will update the API documentation by Friday.",
               owner="Priya", deadline="Friday"),
        action("Send the cost report to the finance team", [4], "I'll send the cost report to the finance team."),
    ],
)


class FakeProvider:
    name = "fake"
    model = "fake-doc-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, system, request):
        self.calls.append({"system": system, "request": request})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, ProviderReply):
            return reply
        return ProviderReply(reply if isinstance(reply, str) else json.dumps(reply), "complete")


def run(out, texts=MEETING):
    return document(refined_transcript(texts), provider=FakeProvider(out), settings=settings())


def rejected_reasons(record, kind):
    return [" | ".join(r.reasons) for r in record.rejected_items if r.kind == kind]


# ---------------------------------------------------------------------------
# Configuration and prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [None, "", "  "])
def test_missing_api_key(key):
    with pytest.raises(DocumentError) as exc:
        document(refined_transcript(), settings=settings(gemini_api_key=key))
    assert exc.value.code == "missing_api_key" and "GEMINI_API_KEY" in exc.value.message


def test_same_model_as_refinement_is_rejected():
    with pytest.raises(DocumentError) as exc:
        document(refined_transcript(), settings=settings(document_model="same", refine_model="same"))
    assert exc.value.code == "config_error" and "REFINE_MODEL" in exc.value.message


def test_empty_model_and_bad_thinking_level():
    for kw in ({"document_model": " "}, {"document_thinking_level": "max"}):
        with pytest.raises(DocumentError) as exc:
            document(refined_transcript(), settings=settings(**kw))
        assert exc.value.code == "config_error"


def test_empty_transcript():
    with pytest.raises(DocumentError) as exc:
        document(refined_transcript(["", "  "]), provider=FakeProvider(GOOD), settings=settings())
    assert exc.value.code == "empty_transcript"


def test_prompt_is_versioned_and_states_the_rules():
    assert PROMPT_PATH.name == f"{PROMPT_VERSION}.txt"
    prompt = load_prompt().lower()
    for phrase in ["authoritative", "do not invent", "discussion", "proposal", "not decisions",
                   "explicit assignment or commitment", "owner", "null", "deadline", "evidence",
                   "copied exactly", "strict json"]:
        assert phrase in prompt, phrase


def test_request_contains_refined_segments_with_ids_and_times():
    provider = FakeProvider(GOOD)
    document(refined_transcript(), provider=provider, settings=settings())
    payload = json.loads(provider.calls[0]["request"].split("\n\n", 1)[1])
    assert payload["segments"][3] == {"segment_id": 3, "start": 15.0, "end": 19.0, "text": MEETING[3]}
    assert provider.calls[0]["system"] == load_prompt()


# ---------------------------------------------------------------------------
# Realistic meeting: everything correct is accepted
# ---------------------------------------------------------------------------


def test_realistic_meeting_record():
    record = run(GOOD)

    assert record.rejected_items == []
    assert record.summary.startswith("The team agreed")                              # 1. summary
    assert [d.evidence_segment_ids for d in record.decisions] == [[1], [5]]           # 2. decisions
    assert "1.29" in record.decisions[0].decision and "October 15" in record.decisions[0].decision
    assert "not" in record.decisions[1].decision                                      # negation kept
    assert [p.evidence_segment_ids for p in record.proposals] == [[2]]                # 3. proposal
    assert len(record.action_items) == 2                                              # 4. actions
    priya, report = record.action_items
    assert (priya.owner, priya.deadline) == ("Priya", "Friday")                       # 5, 7
    assert (report.owner, report.deadline) == (None, None)                            # 6, 8, 9
    assert report.owner_display == report.deadline_display == "Unspecified"
    assert priya.evidence_segment_ids == [3]                                          # 11
    assert record.document_model == "fake-doc-model" and record.prompt_version == PROMPT_VERSION


def test_refined_transcript_is_not_modified():
    refined = refined_transcript()
    before = refined.model_dump_json()
    document(refined, provider=FakeProvider(GOOD), settings=settings())
    assert refined.model_dump_json() == before


def test_quote_matching_ignores_case_and_punctuation():
    record = run(output(decisions=[decision("Do not change the database schema this sprint.", [5],
                                            "we decided NOT to change the database-schema, this sprint")]))
    assert len(record.decisions) == 1


def test_placeholder_owner_and_deadline_become_null():
    record = run(output(actions=[action("Send the cost report", [4], "I'll send the cost report",
                                        owner="Unknown", deadline="TBD")]))
    [a] = record.action_items
    assert a.owner is None and a.deadline is None


# ---------------------------------------------------------------------------
# Decisions vs proposals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item",
    [
        decision("Move the nightly backups to 2 AM.", [2], "Maybe we could also move the nightly backups to 2 AM"),
        decision("Adopt the new on-call rotation.", [9], "Has everyone agreed to the new on-call rotation?"),
        decision("Review the monitoring alerts.", [7], "Someone should review the monitoring alerts."),
        decision("Update the onboarding guide.", [8], "We need to update the onboarding guide."),
    ],
)
def test_unsupported_decision_is_rejected(item):
    record = run(output(decisions=[item]))
    assert record.decisions == []
    assert "no explicit agreement" in rejected_reasons(record, RecordItemKind.DECISION)[0]


def test_proposal_kept_separate_from_decisions():
    record = run(output(proposals=[proposal("Move the nightly backups to 2 AM.", [2],
                                            "move the nightly backups to 2 AM")]))
    assert record.decisions == [] and len(record.proposals) == 1


def test_decision_that_drops_a_negation_is_rejected():
    record = run(output(decisions=[decision("Change the database schema this sprint.", [5],
                                            "We decided not to change the database schema this sprint.")]))
    assert record.decisions == []
    assert "negated" in rejected_reasons(record, RecordItemKind.DECISION)[0]


def test_decision_with_invented_number_is_rejected():
    record = run(output(decisions=[decision("Upgrade Kubernetes to version 1.30.", [1],
                                            "We agreed to upgrade the Kubernetes cluster")]))
    assert "numbers not in the evidence" in rejected_reasons(record, RecordItemKind.DECISION)[0]


def test_decision_with_invented_date_is_rejected():
    record = run(output(decisions=[decision("Upgrade the Kubernetes cluster on Monday.", [1],
                                            "We agreed to upgrade the Kubernetes cluster")]))
    assert "dates not in the evidence" in rejected_reasons(record, RecordItemKind.DECISION)[0]


def test_proposal_with_invented_money_is_rejected():
    record = run(output(proposals=[proposal("Move backups to 2 AM for $500.", [2],
                                            "move the nightly backups to 2 AM")]))
    assert record.proposals == []
    assert "not in the evidence" in rejected_reasons(record, RecordItemKind.PROPOSAL)[0]


# ---------------------------------------------------------------------------
# Action items
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item",
    [
        action("Review the monitoring alerts", [7], "Someone should review the monitoring alerts."),
        action("Update the onboarding guide", [8], "We need to update the onboarding guide."),
        action("Move the nightly backups", [2], "Maybe we could also move the nightly backups to 2 AM"),
    ],
)
def test_discussion_does_not_become_an_action_item(item):
    record = run(output(actions=[item]))
    assert record.action_items == []
    assert "no explicit assignment or commitment" in rejected_reasons(record, RecordItemKind.ACTION_ITEM)[0]


@pytest.mark.parametrize("owner", ["I", "me", "We", "someone", "the team", "Speaker"])
def test_i_will_without_speaker_identity_gets_no_owner(owner):
    record = run(output(actions=[action("Send the cost report", [4], "I'll send the cost report", owner=owner)]))
    assert record.action_items == []
    assert "does not identify a named person" in rejected_reasons(record, RecordItemKind.ACTION_ITEM)[0]


def test_invented_owner_is_rejected():
    record = run(output(actions=[action("Send the cost report", [4], "I'll send the cost report", owner="Rahul")]))
    assert "not named in the evidence" in rejected_reasons(record, RecordItemKind.ACTION_ITEM)[0]


@pytest.mark.parametrize("deadline", ["Monday", "2026-10-10", "end of the month", "tomorrow"])
def test_invented_deadline_is_rejected(deadline):
    record = run(output(actions=[action("Send the cost report", [4], "I'll send the cost report",
                                        deadline=deadline)]))
    assert record.action_items == []
    assert "not stated in the evidence" in rejected_reasons(record, RecordItemKind.ACTION_ITEM)[0]


def test_deadline_wording_from_transcript_is_accepted():
    record = run(output(actions=[action("Update the API documentation", [3], "Priya will update the API documentation",
                                        owner="Priya", deadline="by Friday")]))
    assert record.action_items[0].deadline == "by Friday"


def test_action_item_with_invented_number_is_rejected():
    record = run(output(actions=[action("Send the 3 cost reports", [4], "I'll send the cost report")]))
    assert "numbers not in the evidence" in rejected_reasons(record, RecordItemKind.ACTION_ITEM)[0]


def test_negated_commitment_is_not_an_action_item():
    texts = ["Priya will not update the docs this week."]
    record = run(output(minutes=[], actions=[action("Update the docs", [0], "Priya will not update the docs",
                                                     owner="Priya")]), texts)
    assert record.action_items == []


# ---------------------------------------------------------------------------
# Evidence grounding
# ---------------------------------------------------------------------------


def test_invalid_segment_id_is_rejected():
    record = run(output(decisions=[decision("Upgrade Kubernetes.", [42], "We agreed to upgrade")]))
    assert "do not exist" in rejected_reasons(record, RecordItemKind.DECISION)[0]


def test_fabricated_quote_is_rejected():
    record = run(output(decisions=[decision("Upgrade Kubernetes.", [1], "We unanimously approved the upgrade")]))
    assert "does not appear in the transcript" in rejected_reasons(record, RecordItemKind.DECISION)[0]


def test_quote_from_wrong_segment_is_rejected():
    record = run(output(actions=[action("Update the API documentation", [4],
                                        "Priya will update the API documentation", owner="Priya")]))
    assert "not in the cited segments" in rejected_reasons(record, RecordItemKind.ACTION_ITEM)[0]


def test_rejected_items_keep_the_original_content():
    item = action("Send the cost report", [4], "I'll send the cost report", owner="Rahul")
    record = run(output(actions=[item]))
    [r] = record.rejected_items
    assert r.kind == RecordItemKind.ACTION_ITEM and r.content == item


def test_minutes_need_valid_segments_and_facts():
    record = run(output(minutes=[
        {"topic": "Budget", "discussion": "The budget is $42,000.", "segment_ids": [6]},
        {"topic": "Budget", "discussion": "The budget is $50,000.", "segment_ids": [6]},
        {"topic": "Ghost", "discussion": "Something.", "segment_ids": [99]},
        {"topic": "Kubernetes Upgrade Timing", "discussion": "The upgrade to 1.29 was agreed.", "segment_ids": [1]},
    ]))
    assert [m.discussion for m in record.minutes] == ["The budget is $42,000.", "The upgrade to 1.29 was agreed."]
    assert len(rejected_reasons(record, RecordItemKind.MINUTE)) == 2


def test_summary_with_invented_facts_is_withheld():
    record = run(output(summary="Rahul approved a budget of $90,000."))
    assert record.summary is None
    assert rejected_reasons(record, RecordItemKind.SUMMARY)


# ---------------------------------------------------------------------------
# Malformed output, blocking, truncation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        '{"summary": "x"',
        json.dumps({**GOOD, "attendees": ["Priya"]}),
        json.dumps({k: v for k, v in GOOD.items() if k != "action_items"}),
        json.dumps({**GOOD, "decisions": [{"decision": "x", "evidence_segment_ids": [1]}]}),
        json.dumps({**GOOD, "action_items": [{**GOOD["action_items"][0], "priority": "high"}]}),
    ],
)
def test_malformed_output_retried_then_fails(bad):
    provider = FakeProvider(bad, bad)
    with pytest.raises(DocumentError) as exc:
        document(refined_transcript(), provider=provider, settings=settings())
    assert exc.value.code == "invalid_response" and len(provider.calls) == 2


def test_malformed_output_recovers_on_retry():
    provider = FakeProvider("garbage", GOOD)
    record = document(refined_transcript(), provider=provider, settings=settings())
    assert len(record.decisions) == 2 and len(provider.calls) == 2


@pytest.mark.parametrize("stop,code", [("blocked", "refused"), ("max_tokens", "output_truncated")])
def test_blocked_and_truncated(stop, code):
    with pytest.raises(DocumentError) as exc:
        document(refined_transcript(), provider=FakeProvider(ProviderReply("", stop)), settings=settings())
    assert exc.value.code == code


# ---------------------------------------------------------------------------
# Gemini provider for this stage
# ---------------------------------------------------------------------------


def gemini_response(text):
    part = SimpleNamespace(text=text, thought=False)
    return SimpleNamespace(prompt_feedback=None, candidates=[SimpleNamespace(
        finish_reason=SimpleNamespace(name="STOP"), content=SimpleNamespace(parts=[part]))])


class FakeGeminiClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls = outcomes, []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes[kwargs["model"]]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def gemini_provider(client, **kw):
    return GeminiDocumentProvider.from_settings(settings(**kw), client=client)


def test_gemini_request_shape():
    client = FakeGeminiClient({"doc-model": gemini_response(json.dumps(GOOD))})
    record = document(refined_transcript(), provider=gemini_provider(client, document_thinking_level="high"),
                      settings=settings())
    assert record.document_model == "doc-model" and len(record.action_items) == 2
    [call] = client.calls
    config = call["config"]
    assert call["model"] == "doc-model"
    assert config.system_instruction == load_prompt()
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == OUTPUT_SCHEMA
    assert OUTPUT_SCHEMA["properties"]["action_items"]["items"]["properties"]["owner"] == {"type": ["string", "null"]}
    assert config.thinking_config.thinking_level.value.lower() == "high"
    assert FAKE_KEY not in repr(call)


def test_overloaded_documentation_model_falls_back():
    client = FakeGeminiClient({
        "doc-model": genai_errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "busy"}}),
        "backup": gemini_response(json.dumps(GOOD)),
    })
    record = document(refined_transcript(), provider=gemini_provider(client, document_fallback_models="backup"),
                      settings=settings())
    assert record.document_model == "backup"


def api_error(cls, code, status, message=f"error mentioning {FAKE_KEY}"):
    return cls(code, {"error": {"code": code, "message": message, "status": status}})


@pytest.mark.parametrize(
    "error,code",
    [
        (api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT", "API key not valid"), "auth_failed"),
        (api_error(genai_errors.ClientError, 404, "NOT_FOUND"), "model_not_found"),
        (api_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED"), "rate_limited"),
        (api_error(genai_errors.ServerError, 503, "UNAVAILABLE"), "provider_error"),
        (httpx.ReadTimeout("timed out"), "timeout"),
        (httpx.ConnectError("refused"), "network_error"),
    ],
)
def test_provider_errors_and_key_never_exposed(error, code):
    client = FakeGeminiClient({"doc-model": error})
    with pytest.raises(DocumentError) as exc:
        document(refined_transcript(), provider=gemini_provider(client), settings=settings())
    assert exc.value.code == code
    assert FAKE_KEY not in str(exc.value) and FAKE_KEY not in exc.value.message
    if code == "model_not_found":
        assert "DOCUMENT_MODEL" in exc.value.message
