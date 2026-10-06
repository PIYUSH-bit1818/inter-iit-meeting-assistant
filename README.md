# Meeting Assistant — Inter IIT Bootcamp Phase 2 (ML PS)

Turns a recorded English meeting into a raw transcript, a domain-refined
transcript, and a structured meeting record (summary, minutes, decisions,
action items) through a three-stage pipeline:

1. **Speech-to-text** — Groq-hosted Whisper (`whisper-large-v3`)
2. **Transcript refinement** — Anthropic LLM (model A)
3. **Meeting documentation** — a separate Anthropic LLM (model B)

> Status: **Phase 3 — speech-to-text done.** Uploads are validated, converted
> to 16 kHz mono WAV and transcribed by Groq Whisper into timestamped segments.
> The two LLM stages are not implemented yet.

See [TECHNICAL.md](TECHNICAL.md) for the models and data flow.

## Requirements

- Python 3.11
- FFmpeg on `PATH` (`ffmpeg -version` should work)
- A Groq API key and an Anthropic API key (an Anthropic **API** key from
  console.anthropic.com — a Claude Pro subscription does not provide one)

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
| `ANTHROPIC_API_KEY` | later phases | — | LLM refinement and documentation |

If `GROQ_API_KEY` is missing, transcription fails with a clear
"GROQ_API_KEY is missing" error instead of calling the API.

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

## Tests

```bash
cd backend
pytest
```

## Project layout

```
backend/app/          FastAPI app, settings, schemas, job store
backend/app/pipeline/ audio, stt, refine, document, guards, export stages
backend/tests/        pytest suite
frontend/app.py       Streamlit UI
prompts/              system prompts for the two LLM stages
samples/              sample recording(s) and generated outputs
docs/                 additional documentation
```

## Outputs (per recording)

Raw transcript, refined transcript, meeting minutes, key decisions and action
items, downloadable as Markdown (human-readable) and JSON (machine-readable).
Owners and deadlines that were not stated in the recording are shown as
**Unspecified**.
