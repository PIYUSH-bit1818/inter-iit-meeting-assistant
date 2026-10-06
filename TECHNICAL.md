# Technical Description

## Models and roles

| Stage | Model | Role |
|---|---|---|
| Speech-to-text | **Groq `whisper-large-v3`** (`STT_MODEL`) | Transcribes the uploaded recording into timestamped segments |
| Transcript refinement | **Google Gemini `gemini-3.6-flash`** (`REFINE_MODEL`) | Cleans up the raw transcript: mis-recognised technical terms, acronyms, punctuation, number formatting — without changing meaning |
| Meeting documentation | **Google Gemini `gemini-3-flash-preview`** (`DOCUMENT_MODEL`) | Produces summary, minutes, decisions, proposals and action items from the refined transcript, each item citing its evidence |

## Data flow

```
upload → validate + normalise (ffmpeg) → PreparedAudio (16 kHz mono WAV)
       → Groq Whisper (chunked if needed) → RawTranscript
       → Gemini refinement (strict JSON) → guards → RefinedTranscript
       → Gemini documentation (strict JSON, different model) → grounding checks → MeetingRecord
       → JSON + Markdown exports
```

Contracts are defined in [backend/app/schemas.py](backend/app/schemas.py).
Segment ids stay stable across stages, so each decision and action item cites
the transcript segments it was drawn from.

## Stage 0 — audio validation ([audio.py](backend/app/pipeline/audio.py))

Rejects missing, empty, unsupported, unreadable, video-only, too-short (< 1 s)
and silent files with user-facing messages, then converts the first audio
stream to 16 kHz mono 16-bit PCM WAV — the format Whisper uses internally.

## Stage 1 — speech-to-text ([stt.py](backend/app/pipeline/stt.py))

**Provider and model.** Groq's hosted Whisper. `whisper-large-v3` is selected
because Groq's documentation lists it as the most accurate option
(10.3 % WER vs 12 % for `whisper-large-v3-turbo`); transcription accuracy
on names, terms, numbers and negation carries 20 of the 100 rubric points.
The model is configurable via `STT_MODEL`.

**Request.** `response_format="verbose_json"`, `timestamp_granularities=["segment"]`,
`language="en"`, `temperature=0`. No prompt or vocabulary hints are sent, so
the raw transcript reflects only what the recogniser heard; terminology
correction is left to the separate refinement stage.

**Output.** A `RawTranscript` of `Segment(id, start, end, text)` in
chronological order. Ids are assigned sequentially (0, 1, 2, …) after merging.
Text is kept exactly as returned except for trimming leading/trailing
whitespace. Timestamps are clamped to the recording's length (Whisper
occasionally reports an end time slightly past the audio).

**Chunking.** Groq accepts at most 25 MB per request on the free tier
(100 MB on the dev tier). 16 kHz mono WAV is ~1.9 MB/min, so recordings
longer than ~10 minutes are split:

- Windows are at most 10 minutes and always under 95 % of the byte limit.
- Each window *owns* a contiguous time range and includes 10 s of extra audio
  on each side, so speech crossing a boundary is heard in full by both
  neighbours.
- A segment is kept only from the window that owns its midpoint. This avoids
  duplicated text at boundaries without any text-similarity heuristics.
- Segment times are shifted by the window's start, so they are relative to
  the original recording.

Nothing is truncated: every part of the recording belongs to exactly one window.

**Errors.** All failures surface as `SttError(message, code, retryable)`:

| Code | Cause |
|---|---|
| `missing_api_key` | `GROQ_API_KEY` not set |
| `auth_failed` | key rejected / no access (401, 403) |
| `model_not_found` | `STT_MODEL` unknown (404) |
| `rate_limited` | 429 after SDK retries |
| `request_too_large` | chunk over the limit (checked before sending) or HTTP 413 |
| `provider_error` | Groq 5xx after SDK retries |
| `provider_rejected` | other 4xx |
| `timeout`, `network_error` | request timed out / could not connect |
| `invalid_response`, `unsupported_response` | missing/invalid segments or timestamps, non-JSON reply, text without timestamps |
| `no_speech` | transcript is empty across the whole recording |

The Groq SDK retries transient failures (connection errors, 429, 5xx) up to
three times before an error is raised.

## Stage 2 — transcript refinement ([refine.py](backend/app/pipeline/refine.py), [guards.py](backend/app/pipeline/guards.py))

**Purpose.** Clean up the raw Whisper transcript: fix recognition errors in
technical terms, acronyms and names, punctuation, capitalisation and number
formatting — without changing what was said. This is cleanup, not
summarisation.

**Provider and model.** Google Gemini, **`gemini-3.6-flash`** (`REFINE_MODEL`).
Selection process (October 2026):

1. Listed the models available to our key via the Gemini Models API and kept
   those supporting `generateContent`.
2. Shortlisted the stable (non-preview) Flash models — `gemini-3.5-flash`,
   `3.6-flash`, `3.7-flash`, `3.8-flash` — all listed as "Free of charge" on
   the free tier on Google's pricing page. The moving `gemini-flash-latest`
   alias was avoided so results are reproducible.
3. Sent each one a real request with our exact JSON schema. `gemini-3.6-flash`
   returned valid schema-conforming JSON; the others returned HTTP 503
   ("high demand") repeatedly at the time of testing.

The app uses only the free tier: the key comes from Google AI Studio, no billing
account is required, and nothing in the code enables paid features. Free-tier
caveats: rate limits are lower, and Google may use free-tier content to improve
its products. `REFINE_FALLBACK_MODELS` can list other models to try when the
primary is overloaded.

**Request.** `generateContent` with
`response_mime_type="application/json"` and `response_json_schema` (strict
schema, `additionalProperties: false` everywhere), the versioned system prompt
[prompts/transcript_refinement_v1.txt](prompts/transcript_refinement_v1.txt),
and `thinking_level="low"`. Temperature is left at the default because Google
recommends 1.0 for Gemini 3 models (lower values can cause looping);
faithfulness is enforced by the schema, the prompt and the guards instead.
Segments are sent as `{segment_id, text}` only — timestamps are never sent, so
the model cannot change them. Long transcripts are sent in batches of ~1,500
words, each with the full transcript attached as read-only context so
terminology stays consistent.

**Output.** A separate `RefinedTranscript`; the `RawTranscript` is never
modified. Its `segments` mirror the raw segments one-to-one (same ids,
timestamps, order) and its `refinements` hold, per segment: original text,
refined text, `changed`, status (`unchanged` / `refined` / `fallback`), the
edits with reasons, and for rejected edits the rejected text and the issues.

**Guards.** The model's output is never trusted directly. Each proposed segment
is compared with the raw text after normalising both to the same form
(number words → digits, `$42,000` ↔ "forty-two thousand dollars",
contractions expanded). It is rejected — and the raw text kept — if:

- any number, amount, percentage or version changes, appears or disappears;
- a negation (not, never, no, can't, won't, …) is added or removed;
- a certainty/commitment word changes (might, could, will, agreed, decided,
  maybe, …) — this blocks turning proposals into decisions;
- a date word (month, weekday, today/tomorrow, …) or currency changes;
- a name is replaced by something that does not sound alike;
- a new technical identifier (`S3`, `k8s`, `v2`) appears without support in
  the raw text;
- words are added, or meaningful words removed (fillers and stutters may go);
- a replaced run of words is not a plausible sound-alike correction;
- the rewrite is too large overall, a question becomes a statement, or the
  refined text is empty;
- the model misquotes the original or reports an edit not present in it;
- the segment is missing from, or duplicated in, the model output.

**Errors.** `RefineError(message, code, retryable)`: `missing_api_key`,
`config_error`, `auth_failed`, `model_not_found`, `rate_limited`,
`request_too_large`, `provider_error`, `provider_rejected`, `timeout`,
`network_error`, `refused` (safety block), `output_truncated`, and
`invalid_response` (output still not valid JSON for the schema after one
retry). The SDK retries 408/429/5xx with backoff (`REFINE_RETRY_ATTEMPTS`).
User-facing messages never include provider error text or the API key.

## Stage 3 — meeting documentation ([document.py](backend/app/pipeline/document.py))

**Purpose.** Turn the refined transcript into the meeting record: a summary,
minutes, decisions, proposals (raised but not agreed) and action items. This
is the second, separate LLM stage.

**Provider and model.** Google Gemini, **`gemini-3-flash-preview`**
(`DOCUMENT_MODEL`), with `gemini-3.5-flash-lite` as the default fallback when
the primary is overloaded (`DOCUMENT_FALLBACK_MODELS`). Selection (October 2026):

1. Stronger models were tried first. `gemini-3.1-pro-preview` and
   `gemini-pro-latest` returned HTTP 429 "exceeded your current quota" on the
   free-tier key (Google lists 3.1 Pro as not available on the free tier), and
   `gemini-2.5-pro` / `gemini-2.5-flash` are "no longer available to new users".
   `gemini-3.5/3.7/3.8-flash` returned HTTP 503 "high demand" repeatedly.
2. Of the models that answered, `gemini-3-flash-preview` is the most capable:
   a full Gemini 3 Flash model with thinking (the alternatives were Flash-Lite
   models). Google's pricing page lists it as "Free of charge" on the free tier.
3. A real request with the stage's JSON schema (including nullable
   `owner`/`deadline`) returned valid output.

**Why a different model from refinement.** The two stages do different jobs:
refinement makes many small, local edits and needs speed and consistency
(`gemini-3.6-flash`, low thinking); documentation reasons over the whole
meeting to tell agreement from discussion and assignment from suggestion
(`gemini-3-flash-preview`, medium thinking). Separate models, prompts, schemas,
settings and error messages keep the stages independent; the app refuses to
start the documentation stage if `DOCUMENT_MODEL` equals `REFINE_MODEL`. The
Gemini call mechanics (retries, fallbacks, error mapping) are shared in
[gemini.py](backend/app/pipeline/gemini.py).

**Input.** Only the refined transcript: `{segment_id, start, end, text}` for
each non-empty segment. The raw transcript is not sent and neither transcript
is modified.

**Prompt.** [prompts/meeting_documentation_v1.txt](prompts/meeting_documentation_v1.txt)
states that the transcript is authoritative; nothing may be invented; minutes
describe discussion; decisions require explicit agreement ("could", "maybe",
questions and ideas are not decisions); proposals stay separate; action items
require an explicit assignment or commitment; `owner` and `deadline` are null
unless stated (no speaker labels, so "I" never yields an owner; relative dates
are not converted to calendar dates; no "Unknown"/"TBD" placeholders); every
item needs segment ids and an exact quote; output is strict JSON.

**Output.** Schema-constrained JSON (`response_json_schema`,
`additionalProperties: false`, nullable owner/deadline) parsed into strict
Pydantic models (`extra="forbid"`) and turned into a `MeetingRecord`:
`summary`, `minutes[topic, discussion, segment_ids]`,
`decisions[decision, evidence_segment_ids, evidence_quote]`,
`proposals[proposal, evidence_segment_ids, evidence_quote]`,
`action_items[task, owner|null, deadline|null, evidence_segment_ids, evidence_quote]`,
and `rejected_items[kind, content, reasons]`.

**Grounding checks** ([guards.py](backend/app/pipeline/guards.py)). Each item
is validated deterministically and is either kept unchanged or moved to
`rejected_items` with its reasons — never rewritten:

| Check | Applies to |
|---|---|
| Cited segment ids exist | minutes, decisions, proposals, action items |
| Quote occurs in the cited segments (case/punctuation-insensitive; a quote found only elsewhere is rejected) | decisions, proposals, action items |
| No number, date word, currency, percentage, name or technical identifier absent from the cited segments | all items (summary: whole transcript) |
| Negation consistent between claim and quote ("decided not to…" stays negative) | decisions, proposals, action items |
| Explicit agreement in a cited statement — not a question, not negated ("haven't decided") | decisions |
| Explicit assignment or commitment ("X will", "can you", "I'll"), not negated ("will not") | action items |
| Owner is a person named in the evidence; "I", "we", "someone", "the team" are rejected | action items |
| Deadline wording appears in the evidence (no computed calendar dates) | action items |

If the summary fails its check it is withheld (`summary: null`). Malformed
output is retried once, then the stage fails with `invalid_response`.

**Errors.** `DocumentError` with the same codes as refinement
(`missing_api_key`, `config_error`, `auth_failed`, `model_not_found`,
`rate_limited`, `provider_error`, `timeout`, `network_error`, `refused`,
`output_truncated`, `invalid_response`) plus `empty_transcript`.

**Known limitations.** The grounding checks are lexical: they verify that
every fact an item states is present in its evidence and that the evidence
contains agreement/commitment wording, but cannot fully verify that an owner
named in a multi-sentence quote is the one doing *this* task. A statement like
"We will not change the schema" (a plan with no explicit agreement) is
rejected as a decision by design.

## Orchestration and exports

[runner.py](backend/app/pipeline/runner.py) runs validate → transcribe →
refine → document → export for each uploaded file in a background thread
(`POST /jobs`, `GET /jobs/{id}`), recording each stage's status and message.
A failing stage marks the job failed with a user-facing message, later stages
are skipped, and earlier results are kept. Uploaded and normalised audio is
deleted when the job ends.

[export.py](backend/app/pipeline/export.py) renders from the same objects:
`raw_transcript.txt`, `refined_transcript.txt`, `meeting_record.md`
(human-readable; unstated owners/deadlines shown as "Unspecified"),
`decisions.json`, `action_items.json`, `action_items.csv` and
`meeting_record.json` (complete machine-readable record with both transcripts
and the models used; unstated owners/deadlines are `null`).

## Faithfulness safeguards

- The raw transcript is never modified; refinement produces a separate object
  and any segment edit that fails a guard falls back to the raw text.
- Decisions are kept separate from proposals; a decision needs explicit
  agreement in its cited evidence.
- Every decision, proposal and action item cites segment ids and an exact
  quote; unsupported items are rejected and kept aside for review.
- Owners and deadlines are kept only when stated in the cited segments;
  otherwise they are stored as `null` and shown as "Unspecified".
- JSON and Markdown are rendered from the same object, so they always agree.
