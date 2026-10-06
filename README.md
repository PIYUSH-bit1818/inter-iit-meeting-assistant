# Meeting Assistant — Inter IIT Bootcamp Phase 2 (ML PS)

Turns a recorded English meeting into a raw transcript, a domain-refined
transcript, and a structured meeting record (summary, minutes, decisions,
action items) through a three-stage pipeline:

1. **Speech-to-text** — Groq-hosted Whisper
2. **Transcript refinement** — Anthropic LLM (model A)
3. **Meeting documentation** — a separate Anthropic LLM (model B)

> Status: **Phase 1 — scaffolding.** The API and UI start, data contracts are
> defined and tested; the pipeline stages are not implemented yet.

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
