# Meeting Assistant — Inter IIT Bootcamp Phase 2 (ML PS)

Turns a recorded English meeting into a raw transcript, a domain-refined
transcript, and a structured meeting record (summary, minutes, decisions,
action items) through a three-stage pipeline:

1. **Speech-to-text** — Groq-hosted Whisper (`whisper-large-v3`)
2. **Transcript refinement** — Google Gemini (`gemini-3.6-flash`)
3. **Meeting documentation** — a different Gemini model (`gemini-3-flash-preview`)

> Status: **Phase 5 — meeting documentation done.** The full pipeline runs
> from upload to a validated meeting record (summary, minutes, decisions,
> proposals, action items). Download buttons in the UI come in Phase 6; the
> export files are already generated internally.

See [TECHNICAL.md](TECHNICAL.md) for the models and data flow.

## Requirements

- Python 3.11
- FFmpeg on `PATH` (`ffmpeg -version` should work)
- A Groq API key ([console.groq.com/keys](https://console.groq.com/keys))
- A Gemini API key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)).
  The free tier is enough; no billing account is needed. Note that on the
  free tier Google may use submitted content to improve its products.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r backend/requirements.txt -r frontend/requirements.txt

cp .env.example .env   # then fill in keys and model IDs
```

## Configuration

All settings come from environment variables or a local `.env` file in the
repo root. `.env` is gitignored — never commit it.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | yes | — | Groq key for speech-to-text ([console.groq.com/keys](https://console.groq.com/keys)) |
| `STT_MODEL` | no | `whisper-large-v3` | Groq Whisper model ID |
| `STT_LANGUAGE` | no | `en` | Language hint sent to Whisper |
| `GROQ_MAX_REQUEST_MB` | no | `25` | Per-request upload limit (25 free tier, 100 dev tier) |
| `GEMINI_API_KEY` | yes | — | Gemini key for transcript refinement |
| `REFINE_MODEL` | no | `gemini-3.6-flash` | Gemini model for refinement |
| `REFINE_FALLBACK_MODELS` | no | — | Comma-separated models to try if the primary is overloaded |
| `REFINE_THINKING_LEVEL` | no | `low` | `minimal` / `low` / `medium` / `high` |
| `REFINE_RETRY_ATTEMPTS` | no | `4` | Attempts per model on 408/429/5xx |
| `REFINE_MAX_WORDS_PER_REQUEST` | no | `1500` | Batch size for long transcripts |
| `DOCUMENT_MODEL` | no | `gemini-3-flash-preview` | Gemini model for meeting documentation (must differ from `REFINE_MODEL`) |
| `DOCUMENT_FALLBACK_MODELS` | no | `gemini-3.5-flash-lite` | Models to try if the documentation model is overloaded |
| `DOCUMENT_THINKING_LEVEL` | no | `medium` | `minimal` / `low` / `medium` / `high` |

If a required key is missing, that stage fails with a clear "… is missing"
error instead of calling the API. Leave optional variables commented out
rather than set to an empty value.

## Raw vs refined transcript

The **raw transcript** is exactly what Whisper returned and is never modified.
The **refined transcript** is a separate object with the same segments, ids,
timestamps and order, where Gemini has fixed recognition errors, terminology,
punctuation and number formatting. For every segment the app keeps the raw
text, the refined text, the edits made, and — when an edit was rejected by the
guards — the rejected text and the reasons. The refinement prompt is
[prompts/transcript_refinement_v1.txt](prompts/transcript_refinement_v1.txt).

## Meeting record

A second, different Gemini model reads only the refined transcript (segment
ids, timestamps, text) and writes the meeting record, using
[prompts/meeting_documentation_v1.txt](prompts/meeting_documentation_v1.txt):

- **Summary** and **minutes** (each minute cites its segments).
- **Decisions** — only things explicitly agreed, confirmed or approved.
- **Proposals** — ideas raised but not agreed; never shown as decisions.
- **Action items** — only explicit assignments or commitments. `owner` and
  `deadline` are `null` (shown as **Unspecified**) unless the recording states
  them. With no speaker labels, "I'll do it" never gets an owner.

Every decision, proposal and action item cites segment ids and an exact quote.
Deterministic checks reject any item whose quote is not in the cited segments,
that mentions a name, number, date, amount, owner or deadline the evidence does
not contain, a "decision" without explicit agreement, or a "task" without an
explicit assignment. Rejected items are kept separately for review
("Withheld items" in the UI) and never appear as part of the record.

## Run

Backend (from `backend/`):

```bash
uvicorn app.main:app --reload --port 8000
```

Check http://localhost:8000/health — it reports which settings are configured
(booleans only, never the values).

Frontend (from the repo root, in a second terminal):

```bash
streamlit run frontend/app.py
```

Upload a recording, click **Process recording**, and follow each stage's
status. Results appear in separate tabs: raw transcript, refined transcript
(side by side with the raw text and any rejected edits), summary, minutes,
decisions, proposals, action items and withheld items. Uploaded audio is
deleted after processing.

API: `POST /jobs` (multipart `file`) starts a job; `GET /jobs/{id}` returns its
stage status and results.

## Tests

```bash
cd backend
pytest
```

## Project layout

```
backend/app/          FastAPI app, settings, schemas, job store
backend/app/pipeline/ audio, stt, refine, document, guards, export, gemini, runner
backend/tests/        pytest suite
frontend/app.py       Streamlit UI
prompts/              versioned LLM prompts
samples/              sample recording(s) and generated outputs
docs/                 additional documentation
```

## Outputs (per recording)

Raw transcript, refined transcript, meeting minutes, key decisions and action
items, downloadable as Markdown (human-readable) and JSON (machine-readable).
Owners and deadlines that were not stated in the recording are shown as
**Unspecified**.
