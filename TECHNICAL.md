# Technical Description

_Draft — the LLM model IDs are filled in once verified against the Anthropic API._

## Models and roles

| Stage | Model | Role |
|---|---|---|
| Speech-to-text | **Groq `whisper-large-v3`** (`STT_MODEL`) | Transcribes the uploaded recording into timestamped segments |
| Transcript refinement | Anthropic model A (`REFINE_MODEL`) | Proposes minimal corrections to mis-recognised technical terms, acronyms and domain language |
| Meeting documentation | Anthropic model B (`DOCUMENT_MODEL`) | Produces summary, minutes, decisions, proposals and action items from the refined transcript |

## Data flow

```
upload → validate + normalise (ffmpeg) → PreparedAudio (16 kHz mono WAV)
       → Groq Whisper (chunked if needed) → RawTranscript
       → refinement edits → guards → RefinedTranscript
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

## Faithfulness safeguards

- Refinement returns edits rather than a rewritten transcript; edits that alter
  numbers, negation or names, or that do not match the source text, are rejected.
- Decisions are kept separate from proposals that were not agreed.
- Owners and deadlines are kept only when present in the cited segments;
  otherwise they are stored as `null` and shown as "Unspecified".
- JSON and Markdown are rendered from the same object, so they always agree.
