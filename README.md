# Meeting Assistant — Inter IIT Bootcamp Phase 2 (ML PS)

An AI meeting assistant that turns a recorded English meeting into an accurate
transcript and a trustworthy written record: a raw transcript, a separately
refined transcript, and a meeting summary, minutes, decisions, proposals and
action items — every item backed by a quote from the recording.

## 1. The problem

Meeting notes are only useful if they are right. Speech recognition garbles
technical terms ("Cuba Ernets" for *Kubernetes*), and language models happily
"tidy up" a meeting into decisions nobody agreed to and tasks nobody was
assigned. This project builds a coordinated multi-model pipeline that fixes the
first problem and guards hard against the second: owners and deadlines appear
only when the recording states them, proposals are never presented as
decisions, and anything the model cannot back up with evidence is withheld.

## 2. Architecture

```
 Browser (Streamlit, frontend/app.py)
    │  upload / poll status / download
    ▼
 FastAPI (backend/app/main.py)  POST /jobs · GET /jobs/{id} · GET /jobs/{id}/exports[/{key}]
    │  background thread per job (pipeline/runner.py)
    ▼
 1 Audio validation      ffprobe: type, size, readability, audio stream, duration
 2 Audio normalization   ffmpeg → 16 kHz mono WAV; rejects silence
 3 Speech-to-text        Groq Whisper large-v3 → RawTranscript (timestamped segments)
 4 Refinement   LLM #1   Gemini 3.6 Flash → RefinedTranscript   (+ deterministic edit guards)
 5 Documentation LLM #2  Gemini 3 Flash   → MeetingRecord       (+ deterministic grounding checks)
 6 Export                txt / md / csv / json, all rendered from the same objects
```

Each stage is a separate module with typed Pydantic inputs and outputs
([backend/app/schemas.py](backend/app/schemas.py)). The two LLM stages use
different models, prompts, schemas and settings.

## 3. Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Audio | FFmpeg / ffprobe |
| Speech-to-text | Groq API, Whisper `whisper-large-v3` |
| LLM #1 — transcript refinement | Google Gemini `gemini-3.6-flash` |
| LLM #2 — meeting documentation | Google Gemini `gemini-3-flash-preview` (fallback `gemini-3.5-flash-lite`) |
| Data contracts | Pydantic v2 (strict models, schema-constrained LLM output) |
| Tests | pytest (offline; all providers mocked) |

## 4. Models and their roles

| Stage | Model | Responsibility |
|---|---|---|
| Speech-to-text | Groq **`whisper-large-v3`** | Transcribe the audio into timestamped segments, verbatim. |
| LLM stage 1 — refinement | Gemini **`gemini-3.6-flash`** | *Transcript cleanup only*: fix recognition errors in technical terms, acronyms and names; punctuation; number formatting. Never summarises or changes meaning. |
| LLM stage 2 — documentation | Gemini **`gemini-3-flash-preview`** | *Meeting understanding*: summary, minutes, decisions, proposals and action items, each citing its evidence. |

The two LLM stages are separate operations with separate prompts:
[prompts/transcript_refinement_v1.txt](prompts/transcript_refinement_v1.txt) and
[prompts/meeting_documentation_v1.txt](prompts/meeting_documentation_v1.txt).
[TECHNICAL.md](TECHNICAL.md) explains how each model was selected.

## 5. Installation

Requirements: Python 3.11, FFmpeg on `PATH` (`ffmpeg -version` should work),
and your own free API keys:

- Groq: [console.groq.com/keys](https://console.groq.com/keys)
- Gemini: [aistudio.google.com/apikey](https://aistudio.google.com/apikey). The free tier is
  enough and needs no billing account. On the free tier Google may use
  submitted content to improve its products, so do not upload confidential meetings.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r backend/requirements.txt -r frontend/requirements.txt
cp .env.example .env        # then put YOUR keys in .env
```

Never share or commit `.env` (it is gitignored).

## 6. Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | yes | — | Speech-to-text |
| `GEMINI_API_KEY` | yes | — | Both LLM stages |
| `STT_MODEL` | no | `whisper-large-v3` | Groq Whisper model |
| `STT_LANGUAGE` | no | `en` | Language hint for Whisper |
| `GROQ_MAX_REQUEST_MB` | no | `25` | Per-request upload limit (25 free tier, 100 dev tier); longer audio is chunked |
| `REFINE_MODEL` | no | `gemini-3.6-flash` | LLM stage 1 |
| `REFINE_FALLBACK_MODELS` | no | — | Models to try if stage 1's model is overloaded |
| `REFINE_THINKING_LEVEL` | no | `low` | `minimal` / `low` / `medium` / `high` |
| `DOCUMENT_MODEL` | no | `gemini-3-flash-preview` | LLM stage 2 (must differ from `REFINE_MODEL`) |
| `DOCUMENT_FALLBACK_MODELS` | no | `gemini-3.5-flash-lite` | Models to try if stage 2's model is overloaded |
| `DOCUMENT_THINKING_LEVEL` | no | `medium` | `minimal` / `low` / `medium` / `high` |
| `MAX_UPLOAD_MB` | no | `100` | Maximum upload size |
| `BACKEND_URL` | no | `http://localhost:8000` | Where the Streamlit app finds the backend |

Leave optional variables commented out rather than set to an empty value.

## 7. How to run

Terminal 1 — backend (from `backend/`):

```bash
uvicorn app.main:app --port 8000
```

Terminal 2 — frontend (from the repo root):

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501. The sidebar shows whether the backend is reachable
and which keys are configured (yes/no only; values are never shown).

## 8. Using the application

1. Choose a recording (WAV, MP3, M4A, OGG, FLAC, WEBM or MP4). The selected file name is shown.
2. Click **Process recording**.
3. Watch the six stages: validation → normalization → transcription →
   refinement → documentation → export. Each shows *running*, *done*,
   *failed* or *skipped* from the backend's real state, with a short result
   (e.g. "18 segments", "2 decisions, 2 proposals, 2 action items").
4. Explore the tabs: **Raw transcript**, **Refined transcript**, **Raw vs
   refined**, **Summary**, **Minutes**, **Decisions**, **Proposals / not
   agreed**, **Action items**, **Withheld / rejected**, **Downloads**.

If something goes wrong (unsupported, empty, corrupted, silent or too-short
file; no speech; missing key; provider rate limit or outage; malformed model
output), the failing stage is marked and a plain-language message is shown.
Earlier results stay available: for example, the raw transcript can still be
downloaded if refinement fails. Uploaded audio is deleted after processing.

## 9. Pipeline

1. **Validation** rejects missing, empty, unsupported, unreadable, video-only
   and too-short (< 1 s) files.
2. **Normalization** converts the first audio stream to 16 kHz mono 16-bit WAV
   and rejects silent recordings.
3. **Speech-to-text** uses Whisper large-v3 via Groq with segment timestamps.
   Recordings longer than about 10 minutes are split into overlapping windows
   and stitched back without duplicated or lost text; timestamps stay relative
   to the original recording. A transcript without any words is reported as
   "no speech".
4. **Refinement (LLM #1)** produces a *separate* refined transcript (see below).
5. **Documentation (LLM #2)** produces the meeting record from the refined
   transcript only.
6. **Export** builds the download files.

## 10. Raw vs refined transcript

- The **raw transcript** is exactly what Whisper returned. It is never modified.
- The **refined transcript** is a separate object with the *same* segment ids,
  timestamps and order, in which recognition errors and formatting have been
  fixed (e.g. "Cuba Ernets upgrade" → "Kubernetes upgrade", "2am" → "2 a.m.").
- The **Raw vs refined** tab shows both side by side for each segment, with
  the edits made and any edit the safety checks rejected (raw text kept).

## 11. Grounding and safety

- **Refinement guards.** Each refined segment is compared with the raw text.
  It is rejected (and the raw text kept) if any number, amount, date,
  percentage, negation, certainty word (*might / could / will / agreed*), name
  or technical identifier changes, if words are added or meaningful words
  removed, or if the rewrite is too large.
- **Evidence for every claim.** Each decision, proposal and action item cites
  segment ids and an exact quote. It is withheld if the ids don't exist, the
  quote isn't in those segments, or the item mentions a name, number, date,
  amount, owner or deadline that the evidence doesn't contain.
- **Proposal ≠ decision.** A decision needs explicit agreement in a statement
  ("we agreed", "we decided"). "Maybe we could…", ideas and questions such as
  "Has everyone agreed?" are not decisions.
- **Discussion ≠ action item.** An action item needs an explicit assignment or
  commitment ("Rahul, can you…", "I'll…"). "Someone should…" or "we need to…"
  is not a task.
- **Owner and deadline are null unless stated.** There are no speaker labels,
  so "I'll do it" never gets an owner. Deadlines use the meeting's own words
  ("Friday") and are never computed. The UI shows "Not specified".
- **Withheld items are shown, not hidden.** Anything that failed a check
  appears under **Withheld / rejected** with the reasons, and is excluded from
  the record and from the decision and task exports.

## 12. Exports

| File | Format | Contents |
|---|---|---|
| `raw_transcript.txt` | text | Timestamped raw segments |
| `refined_transcript.txt` | text | Timestamped refined segments |
| `meeting_minutes.md` | Markdown | Summary and minutes with segment references and timestamps |
| `decisions.md`, `decisions.json` | Markdown, JSON | Decisions (and proposals in the Markdown) with evidence |
| `action_items.csv`, `action_items.json` | CSV, JSON | Task, owner, deadline, evidence |
| `meeting_record.md` | Markdown | The complete human-readable record |
| `meeting_record.json` | JSON | Everything: raw transcript, refined transcript (with per-segment edits), summary, minutes, decisions, proposals, action items, withheld items, models used |

The backend generates every file from the same objects the UI displays, so the
UI, Markdown, CSV and JSON never disagree. Unstated owners and deadlines are
`null` in JSON and "Not specified" in human-readable files.

## 13. Testing

```bash
cd backend
pytest
```

334 automated tests cover audio validation and normalization (with real
ffmpeg-generated fixtures), speech-to-text and chunking, refinement and its
guards, documentation and its grounding checks, the pipeline runner, the API
and the exports. All external services are mocked, so the tests need no API
keys and make no network calls.

## 14. Demo

A reproducible 1.7-minute, multi-voice demo meeting can be generated on
Windows with the built-in speech voices:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make_demo_recording.ps1
```

It writes `data/demo/demo_meeting.wav` (gitignored). The meeting contains:

- a confirmed decision and a negative decision ("decided not to change the database schema");
- two proposals and a question that is not a decision;
- an action item with owner and deadline ("Rahul, can you update the API documentation by Friday?");
- an action item with neither ("I'll send the cost report");
- statements that are not tasks, plus numbers, a date, money, a percentage and technical terms.

Any other English meeting recording can be used instead.

Suggested demo flow: upload the file → show the six stages completing →
compare raw vs refined ("Cuba Ernets" → "Kubernetes") → decisions vs
proposals → action items with "Not specified" → downloads → open
`meeting_record.json`.

## 15. Known limitations

- No speaker diarization: speakers are not identified, so "I" or "we" never
  become owners (by design) and minutes do not attribute statements.
- The grounding checks are lexical. They verify that every stated fact appears
  in the cited evidence and that agreement or commitment wording is present,
  but cannot fully prove that a named owner in a multi-sentence quote is the
  one doing *that* task.
- The decision rule is strict: a plan stated without explicit agreement ("we
  will not change the schema") is not recorded as a decision.
- Free-tier limits apply: Gemini and Groq rate limits and occasional "model
  overloaded" responses are handled with retries and a fallback model for
  documentation. The documentation model is a preview model.
- Jobs are kept in memory (the most recent 50); restarting the backend clears them.
- Whisper can mishear words; refinement only fixes errors that the context makes clear.

## 16. Project layout

```
backend/app/main.py           FastAPI endpoints
backend/app/config.py         settings (.env)
backend/app/schemas.py        Pydantic contracts between stages
backend/app/jobs.py           in-memory job store
backend/app/pipeline/         audio, stt, refine, document, guards, gemini, export, runner
backend/tests/                pytest suite
frontend/app.py               Streamlit UI
prompts/                      versioned LLM prompts
scripts/                      demo recording generator
TECHNICAL.md                  technical description
```
