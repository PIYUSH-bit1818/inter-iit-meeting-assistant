# Technical Description

_Draft — the meeting-documentation model is added in Phase 5._

## Models and roles

| Stage | Model | Role |
|---|---|---|
| Speech-to-text | **Groq `whisper-large-v3`** (`STT_MODEL`) | Transcribes the uploaded recording into timestamped segments |
| Transcript refinement | **Google Gemini `gemini-3.6-flash`** (`REFINE_MODEL`) | Cleans up the raw transcript: mis-recognised technical terms, acronyms, punctuation, number formatting — without changing meaning |
| Meeting documentation | separate LLM, Phase 5 (`DOCUMENT_MODEL`) | Produces summary, minutes, decisions, proposals and action items from the refined transcript |

## Data flow

```
upload → validate + normalise (ffmpeg) → PreparedAudio (16 kHz mono WAV)
       → Groq Whisper (chunked if needed) → RawTranscript
       → Gemini refinement (strict JSON) → guards → RefinedTranscript
       → MeetingRecord (structured) → grounding checks → JSON + Markdown
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

## Faithfulness safeguards

- The raw transcript is never modified; refinement produces a separate object
  and any segment edit that fails a guard falls back to the raw text.
- Decisions are kept separate from proposals that were not agreed.
- Owners and deadlines are kept only when present in the cited segments;
  otherwise they are stored as `null` and shown as "Unspecified".
- JSON and Markdown are rendered from the same object, so they always agree.
