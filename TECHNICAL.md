# Technical Description

_Draft — model IDs are filled in once verified against the provider APIs._

## Models and roles

| Stage | Model | Role |
|---|---|---|
| Speech-to-text | Whisper via Groq API (`STT_MODEL`) | Transcribes the uploaded recording into timestamped segments |
| Transcript refinement | Anthropic model A (`REFINE_MODEL`) | Proposes minimal corrections to mis-recognised technical terms, acronyms and domain language |
| Meeting documentation | Anthropic model B (`DOCUMENT_MODEL`) | Produces summary, minutes, decisions, proposals and action items from the refined transcript |

## Data flow

```
upload → validate + normalise (ffmpeg) → RawTranscript
       → refinement edits → guards → RefinedTranscript
       → MeetingRecord (structured) → grounding checks → JSON + Markdown
```

Contracts are defined in [backend/app/schemas.py](backend/app/schemas.py).
Segment ids stay stable across stages, so each decision and action item cites
the transcript segments it was drawn from.

## Faithfulness safeguards

- Refinement returns edits rather than a rewritten transcript; edits that alter
  numbers, negation or names, or that do not match the source text, are rejected.
- Decisions are kept separate from proposals that were not agreed.
- Owners and deadlines are kept only when present in the cited segments;
  otherwise they are stored as `null` and shown as "Unspecified".
- JSON and Markdown are rendered from the same object, so they always agree.
