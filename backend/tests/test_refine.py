"""Tests for the refinement stage. No network and no real API key: the
provider (or the Gemini client) is replaced with fakes."""

import json
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from app.config import Settings
from app.pipeline.refine import (
    OUTPUT_SCHEMA,
    PROMPT_PATH,
    PROMPT_VERSION,
    GeminiRefineProvider,
    ProviderReply,
    RefineError,
    load_prompt,
    refine,
)
from app.schemas import RawTranscript, RefinementStatus, Segment

FAKE_KEY = "AIzaFAKE-test-0123456789-secret"


def settings(**kw) -> Settings:
    base = {"gemini_api_key": FAKE_KEY, "refine_model": "test-model", "_env_file": None}
    return Settings(**{**base, **kw})


def raw_transcript(*texts: str) -> RawTranscript:
    segs = [Segment(id=i, start=i * 3.0, end=i * 3.0 + 2.5, text=t) for i, t in enumerate(texts)]
    return RawTranscript(segments=segs, stt_model="whisper-large-v3", duration_seconds=len(texts) * 3.0)


def entry(segment_id, original, refined, changes=()):
    return {
        "segment_id": segment_id,
        "original_text": original,
        "refined_text": refined,
        "changed": refined != original,
        "changes": [{"original_span": o, "refined_span": r, "reason": "terminology"} for o, r in changes],
    }


class FakeProvider:
    """Answers like the model would, from a raw-text -> refined-text mapping.

    ``replies`` (if given) are returned verbatim, one per call, instead.
    """

    name = "fake"
    model = "fake-refiner"

    def __init__(self, mapping=None, replies=None, transform=None):
        self.mapping = mapping or {}
        self.replies = list(replies or [])
        self.transform = transform
        self.calls: list[dict] = []

    def complete(self, system, context, request):
        payload = json.loads(request[request.index("{"):])
        self.calls.append({"system": system, "context": context, "payload": payload})
        if self.replies:
            reply = self.replies.pop(0)
            return reply if isinstance(reply, ProviderReply) else ProviderReply(reply, "complete")
        items = [
            entry(s["segment_id"], s["text"], self.mapping.get(s["text"], s["text"]))
            for s in payload["segments"]
        ]
        if self.transform:
            items = self.transform(items)
        return ProviderReply(json.dumps({"segments": items}), "complete")


def run(raw, provider, **settings_kw):
    return refine(raw, provider=provider, settings=settings(**settings_kw))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_api_key(key):
    with pytest.raises(RefineError) as exc:
        refine(raw_transcript("hello"), settings=settings(gemini_api_key=key))
    assert exc.value.code == "missing_api_key"
    assert "GEMINI_API_KEY" in exc.value.message


def test_missing_model():
    with pytest.raises(RefineError) as exc:
        refine(raw_transcript("hello"), settings=settings(refine_model="  "))
    assert exc.value.code == "config_error" and "REFINE_MODEL" in exc.value.message


def test_invalid_thinking_level():
    with pytest.raises(RefineError) as exc:
        refine(raw_transcript("hello"), settings=settings(refine_thinking_level="extreme"))
    assert exc.value.code == "config_error"


def test_prompt_file_is_versioned_and_loaded():
    assert PROMPT_PATH.name == f"{PROMPT_VERSION}.txt" and PROMPT_PATH.is_file()
    prompt = load_prompt()
    for phrase in ["authoritative", "not summarization", "Negation", "proposal", "owner",
                   "number", "date", "Names", "identifier", "smallest edit", "segment_id"]:
        assert phrase.lower() in prompt.lower(), phrase


# ---------------------------------------------------------------------------
# Successful refinement
# ---------------------------------------------------------------------------


def test_successful_refinement_with_terminology_correction():
    raw = raw_transcript("we deploy on cubernetties")
    provider = FakeProvider(transform=lambda items: [
        entry(0, items[0]["original_text"], "We deploy on Kubernetes.", [("cubernetties", "Kubernetes")])
    ])
    result = run(raw, provider)

    [r] = result.refinements
    assert r.status == RefinementStatus.REFINED and r.changed
    assert r.original_text == "we deploy on cubernetties"
    assert r.refined_text == result.segments[0].text == "We deploy on Kubernetes."
    assert [(c.original_span, c.refined_span) for c in r.changes] == [("cubernetties", "Kubernetes")]
    assert result.refine_model == "fake-refiner" and result.prompt_version == PROMPT_VERSION


def test_unchanged_segment():
    result = run(raw_transcript("Good morning everyone."), FakeProvider())
    [r] = result.refinements
    assert r.status == RefinementStatus.UNCHANGED and not r.changed and r.changes == []
    assert result.segments[0].text == "Good morning everyone."


def test_acronym_correction():
    result = run(raw_transcript("update the a p i docs"),
                 FakeProvider({"update the a p i docs": "Update the API docs."}))
    assert result.refinements[0].status == RefinementStatus.REFINED
    assert result.segments[0].text == "Update the API docs."


def test_changes_derived_when_model_lists_none():
    result = run(raw_transcript("deploy on cubernetties"),
                 FakeProvider({"deploy on cubernetties": "Deploy on Kubernetes."}))
    changes = result.refinements[0].changes
    assert changes and any(c.refined_span.startswith("Kubernetes") for c in changes)


def test_multiple_segments_ids_timestamps_and_order_preserved():
    texts = ["um so the post gres migration", "Priya will review it.", "we use react for the front end"]
    mapping = {texts[0]: "So the Postgres migration.", texts[2]: "We use React for the front end."}
    # Model returns its entries in reverse order - output must follow the raw order
    provider = FakeProvider(mapping, transform=lambda items: list(reversed(items)))
    raw = raw_transcript(*texts)
    result = run(raw, provider)

    assert [s.id for s in result.segments] == [0, 1, 2]
    assert [(s.start, s.end) for s in result.segments] == [(s.start, s.end) for s in raw.segments]
    assert [r.status for r in result.refinements] == [
        RefinementStatus.REFINED, RefinementStatus.UNCHANGED, RefinementStatus.REFINED]
    assert [s.text for s in result.segments] == [
        "So the Postgres migration.", "Priya will review it.", "We use React for the front end."]


def test_exact_raw_to_refined_relationship():
    raw = raw_transcript("we use cubernetties", "The budget is forty-two thousand dollars.")
    provider = FakeProvider({"we use cubernetties": "We use Kubernetes.",
                             "The budget is forty-two thousand dollars.": "The budget is $42,000."})
    result = run(raw, provider)
    for raw_seg, ref_seg, r in zip(raw.segments, result.segments, result.refinements):
        assert r.segment_id == raw_seg.id == ref_seg.id
        assert (r.start, r.end) == (raw_seg.start, raw_seg.end) == (ref_seg.start, ref_seg.end)
        assert r.original_text == raw_seg.text
        assert r.refined_text == ref_seg.text


def test_raw_transcript_is_never_modified():
    raw = raw_transcript("we will not ship", "we use cubernetties")
    before = raw.model_dump()
    run(raw, FakeProvider({"we will not ship": "We will ship.", "we use cubernetties": "We use Kubernetes."}))
    assert raw.model_dump() == before


def test_timestamps_are_never_sent_to_the_model():
    provider = FakeProvider()
    run(raw_transcript("hello there", "general kenobi"), provider)
    sent = provider.calls[0]["payload"]["segments"]
    assert sent == [{"segment_id": 0, "text": "hello there"}, {"segment_id": 1, "text": "general kenobi"}]


def test_blank_raw_segment_is_not_sent_and_kept():
    raw = raw_transcript("hello there", "   ")
    provider = FakeProvider()
    result = run(raw, provider)
    assert [s["segment_id"] for s in provider.calls[0]["payload"]["segments"]] == [0]
    assert result.refinements[1].status == RefinementStatus.UNCHANGED
    assert result.segments[1].text == "   "


# ---------------------------------------------------------------------------
# Unsafe refinements fall back to the raw text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_text,bad,expected",
    [
        ("The budget is forty-two thousand dollars.", "The budget is $24,000.", "numbers changed"),
        ("We will not change the database schema.", "We will change the database schema.", "negation changed"),
        ("We ship next week.", "We ship next week, on October 10.", "dates changed"),
        ("It costs a lot.", "It costs $5,000.", "numbers changed"),
        ("It costs 50 rupees.", "It costs 50 dollars.", "monetary units changed"),
        ("Let's discuss the roadmap.", "The team reviewed and finalized the roadmap for next quarter.", "rewrite too large"),
        ("We might migrate to Kubernetes next month.", "We agreed to migrate to Kubernetes next month.", "certainty"),
        ("Priya could update the API documentation.", "Priya will update the API documentation.", "certainty"),
        ("Ask Priya about it.", "Ask Maria about it.", "name changed"),
        ("Okay, let's start.", "", "refined text is empty"),
    ],
)
def test_unsafe_refinement_falls_back_to_raw(raw_text, bad, expected):
    result = run(raw_transcript(raw_text), FakeProvider({raw_text: bad}))
    [r] = result.refinements
    assert r.status == RefinementStatus.FALLBACK
    assert r.refined_text == r.original_text == raw_text
    assert result.segments[0].text == raw_text
    assert r.rejected_text == bad.strip()
    assert any(expected in issue for issue in r.issues), r.issues


def test_one_bad_segment_does_not_affect_others():
    raw = raw_transcript("we use cubernetties", "We will not ship.")
    result = run(raw, FakeProvider({"we use cubernetties": "We use Kubernetes.", "We will not ship.": "We will ship."}))
    assert [r.status for r in result.refinements] == [RefinementStatus.REFINED, RefinementStatus.FALLBACK]


def test_misquoted_original_is_rejected():
    def transform(items):
        items[0]["original_text"] = "something else entirely"
        items[0]["refined_text"] = "We use Kubernetes."
        return items

    result = run(raw_transcript("we use cubernetties"), FakeProvider(transform=transform))
    assert result.refinements[0].status == RefinementStatus.FALLBACK
    assert "misquoted" in result.refinements[0].issues[0]


def test_segment_missing_from_output_falls_back():
    result = run(raw_transcript("a b c", "d e f"), FakeProvider(transform=lambda items: items[:1]))
    r = result.refinements[1]
    assert r.status == RefinementStatus.FALLBACK and "missing" in r.issues[0]
    assert result.segments[1].text == "d e f"


def test_duplicated_segment_falls_back():
    result = run(raw_transcript("we use cubernetties"),
                 FakeProvider(transform=lambda items: [
                     entry(0, "we use cubernetties", "We use Kubernetes."),
                     entry(0, "we use cubernetties", "We use Kubernetes!")]))
    assert result.refinements[0].status == RefinementStatus.FALLBACK


def test_extra_segments_from_model_are_ignored():
    result = run(raw_transcript("hello"),
                 FakeProvider(transform=lambda items: items + [entry(7, "invented", "Invented segment.")]))
    assert [s.id for s in result.segments] == [0]
    assert result.segments[0].text == "hello"


# ---------------------------------------------------------------------------
# Malformed output, refusals, truncation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_reply",
    [
        "not json at all",
        '{"segments": [',
        "[]",
        '{"items": []}',
        '{"segments": [{"segment_id": 0, "refined_text": "x"}]}',
        '{"segments": [{"segment_id": "zero", "original_text": "a", "refined_text": "a", "changed": false, "changes": []}]}',
        json.dumps({"segments": [{**entry(0, "hello", "hello"), "speaker": "Priya"}]}),
        json.dumps({"segments": [], "summary": "extra top-level field"}),
    ],
)
def test_malformed_output_retried_then_fails(bad_reply):
    provider = FakeProvider(replies=[bad_reply, bad_reply])
    with pytest.raises(RefineError) as exc:
        run(raw_transcript("hello"), provider)
    assert exc.value.code == "invalid_response"
    assert len(provider.calls) == 2


def test_malformed_output_recovers_on_retry():
    good = json.dumps({"segments": [entry(0, "hello", "Hello.")]})
    provider = FakeProvider(replies=["garbage", good])
    result = run(raw_transcript("hello"), provider)
    assert result.segments[0].text == "Hello." and len(provider.calls) == 2


def test_refusal():
    provider = FakeProvider(replies=[ProviderReply("", "blocked")])
    with pytest.raises(RefineError) as exc:
        run(raw_transcript("hello"), provider)
    assert exc.value.code == "refused"


def test_truncated_output_splits_batch():
    provider = FakeProvider()
    original = provider.complete
    state = {"first": True}

    def complete(system, context, request):
        if state.pop("first", False):
            return ProviderReply('{"segments": [', "max_tokens")
        return original(system, context, request)

    provider.complete = complete
    result = run(raw_transcript("one", "two", "three", "four"), provider)
    assert [s.text for s in result.segments] == ["one", "two", "three", "four"]
    assert [len(c["payload"]["segments"]) for c in provider.calls] == [2, 2]


def test_truncated_single_segment_fails():
    provider = FakeProvider(replies=[ProviderReply("", "max_tokens")])
    with pytest.raises(RefineError) as exc:
        run(raw_transcript("hello"), provider)
    assert exc.value.code == "output_truncated"


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_long_transcripts_are_batched_with_shared_context():
    texts = [f"segment number {i} says something" for i in range(10)]  # 5 words each
    provider = FakeProvider()
    result = run(raw_transcript(*texts), provider, refine_max_words_per_request=12)

    assert len(provider.calls) == 5
    assert [len(c["payload"]["segments"]) for c in provider.calls] == [2] * 5
    contexts = {c["context"] for c in provider.calls}
    assert len(contexts) == 1, "every batch gets the same (cacheable) context"
    assert "[9] segment number 9" in contexts.pop()
    assert [s.id for s in result.segments] == list(range(10))


def test_single_batch_has_no_context():
    provider = FakeProvider()
    run(raw_transcript("short", "meeting"), provider)
    assert provider.calls[0]["context"] is None


# ---------------------------------------------------------------------------
# Realistic example
# ---------------------------------------------------------------------------


def test_realistic_meeting_meaning_is_preserved():
    texts = [
        "okay so um we need to move the cube nettees cluster to version one point two nine",
        "Priya can you update the a p i documentation by Friday October tenth",
        "the budget for this is forty two thousand dollars",
        "we will not change the database schema this sprint",
        "maybe we could also look at post gres for the analytics",
    ]
    good = {
        texts[0]: "Okay, so we need to move the Kubernetes cluster to version 1.29.",
        texts[1]: "Priya, can you update the API documentation by Friday, October 10?",
        texts[2]: "The budget for this is $42,000.",
        texts[3]: "We will not change the database schema this sprint.",
        # Unsafe: drops the hedge and turns a suggestion into a decision
        texts[4]: "We will also use Postgres for the analytics.",
    }
    raw = raw_transcript(*texts)
    result = run(raw, FakeProvider(good))
    status = [r.status for r in result.refinements]

    assert status[:4] == [RefinementStatus.REFINED] * 4
    assert status[4] == RefinementStatus.FALLBACK
    assert result.segments[4].text == texts[4], "proposal must stay a proposal"
    assert "Kubernetes" in result.segments[0].text and "1.29" in result.segments[0].text
    assert "Priya" in result.segments[1].text and "October 10" in result.segments[1].text
    assert result.segments[1].text.endswith("?"), "question stays a question"
    assert "$42,000" in result.segments[2].text
    assert "not" in result.segments[3].text


# ---------------------------------------------------------------------------
# Gemini provider: request shape, response handling and error mapping
# ---------------------------------------------------------------------------


def gemini_response(text="", finish="STOP", block_reason=None, thought_text=None):
    parts = []
    if thought_text is not None:
        parts.append(SimpleNamespace(text=thought_text, thought=True))
    parts.append(SimpleNamespace(text=text, thought=False))
    candidate = SimpleNamespace(finish_reason=SimpleNamespace(name=finish),
                                content=SimpleNamespace(parts=parts))
    return SimpleNamespace(
        prompt_feedback=SimpleNamespace(block_reason=block_reason) if block_reason else None,
        candidates=[] if block_reason else [candidate],
    )


class FakeGeminiClient:
    """``outcomes`` maps model name -> response object or exception."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls: list[dict] = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes[kwargs["model"]]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def gemini_provider(client, **kw):
    return GeminiRefineProvider.from_settings(settings(**kw), client=client)


GOOD = json.dumps({"segments": [entry(0, "we use cubernetties", "We use Kubernetes.")]})


def test_gemini_request_shape():
    client = FakeGeminiClient({"test-model": gemini_response(GOOD)})
    provider = gemini_provider(client, refine_thinking_level="medium")
    result = refine(raw_transcript("we use cubernetties"), provider=provider, settings=settings())

    assert result.segments[0].text == "We use Kubernetes."
    assert result.refine_model == "test-model"
    [call] = client.calls
    assert call["model"] == "test-model"
    config = call["config"]
    assert config.system_instruction == load_prompt()
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == OUTPUT_SCHEMA
    assert config.thinking_config.thinking_level.value.lower() == "medium"
    assert config.temperature is None  # Gemini 3 default (1.0) is recommended
    assert config.automatic_function_calling.disable is True
    assert FAKE_KEY not in repr(call)


def test_gemini_ignores_thought_parts():
    client = FakeGeminiClient({"test-model": gemini_response(GOOD, thought_text="internal notes {")})
    result = refine(raw_transcript("we use cubernetties"), provider=gemini_provider(client),
                    settings=settings())
    assert result.segments[0].text == "We use Kubernetes."


def test_gemini_context_sent_as_separate_part():
    texts = [f"segment number {i} says something" for i in range(4)]
    replies = iter([
        json.dumps({"segments": [entry(i, texts[i], texts[i]) for i in (0, 1)]}),
        json.dumps({"segments": [entry(i, texts[i], texts[i]) for i in (2, 3)]}),
    ])

    class Client(FakeGeminiClient):
        def _generate(self, **kwargs):
            self.calls.append(kwargs)
            return gemini_response(next(replies))

    client = Client({})
    refine(raw_transcript(*texts), provider=gemini_provider(client),
           settings=settings(refine_max_words_per_request=10))
    assert len(client.calls) == 2
    for call in client.calls:
        context, request = call["contents"]
        assert context.startswith("Full meeting transcript") and "[3] segment number 3" in context
        assert request.startswith("Refine these transcript segments")


@pytest.mark.parametrize(
    "response,code",
    [
        (gemini_response(finish="SAFETY"), "refused"),
        (gemini_response(finish="PROHIBITED_CONTENT"), "refused"),
        (gemini_response(block_reason="SAFETY"), "refused"),
        (gemini_response(finish="MAX_TOKENS"), "output_truncated"),
        (gemini_response("not json"), "invalid_response"),
    ],
)
def test_gemini_finish_reasons(response, code):
    client = FakeGeminiClient({"test-model": response})
    with pytest.raises(RefineError) as exc:
        refine(raw_transcript("hello"), provider=gemini_provider(client), settings=settings())
    assert exc.value.code == code


def api_error(cls, code, status, message=f"error mentioning {FAKE_KEY}"):
    return cls(code, {"error": {"code": code, "message": message, "status": status}})


@pytest.mark.parametrize(
    "error,code,retryable",
    [
        (api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT",
                   f"API key not valid. Key {FAKE_KEY}"), "auth_failed", False),
        (api_error(genai_errors.ClientError, 403, "PERMISSION_DENIED"), "auth_failed", False),
        (api_error(genai_errors.ClientError, 404, "NOT_FOUND"), "model_not_found", False),
        (api_error(genai_errors.ClientError, 413, "REQUEST_TOO_LARGE"), "request_too_large", False),
        (api_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED"), "rate_limited", True),
        (api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT"), "provider_rejected", False),
        (api_error(genai_errors.ServerError, 500, "INTERNAL"), "provider_error", True),
        (api_error(genai_errors.ServerError, 503, "UNAVAILABLE"), "provider_error", True),
        (httpx.ReadTimeout("timed out"), "timeout", True),
        (httpx.ConnectError("connection refused"), "network_error", True),
    ],
)
def test_provider_failures_and_key_never_exposed(error, code, retryable):
    client = FakeGeminiClient({"test-model": error})
    with pytest.raises(RefineError) as exc:
        refine(raw_transcript("hello"), provider=gemini_provider(client), settings=settings())
    assert exc.value.code == code and exc.value.retryable is retryable
    assert FAKE_KEY not in exc.value.message
    assert FAKE_KEY not in str(exc.value)


def test_overloaded_model_falls_back_to_next_model():
    client = FakeGeminiClient({
        "test-model": api_error(genai_errors.ServerError, 503, "UNAVAILABLE"),
        "backup-model": gemini_response(GOOD),
    })
    provider = gemini_provider(client, refine_fallback_models=" backup-model , ")
    result = refine(raw_transcript("we use cubernetties"), provider=provider, settings=settings())
    assert [c["model"] for c in client.calls] == ["test-model", "backup-model"]
    assert result.refine_model == "backup-model", "records the model that actually answered"
    assert result.segments[0].text == "We use Kubernetes."


def test_non_transient_error_does_not_try_fallback():
    client = FakeGeminiClient({
        "test-model": api_error(genai_errors.ClientError, 404, "NOT_FOUND"),
        "backup-model": gemini_response(GOOD),
    })
    with pytest.raises(RefineError) as exc:
        refine(raw_transcript("hello"), provider=gemini_provider(client, refine_fallback_models="backup-model"),
               settings=settings())
    assert exc.value.code == "model_not_found"
    assert [c["model"] for c in client.calls] == ["test-model"]


def test_key_not_exposed_in_config_errors():
    with pytest.raises(RefineError) as exc:
        refine(raw_transcript("hello"), settings=settings(refine_model="  "))
    assert FAKE_KEY not in str(exc.value)
