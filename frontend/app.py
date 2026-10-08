"""Scripted - Streamlit UI for the AI meeting assistant.

Run from the repo root (with the backend running): streamlit run frontend/app.py

Everything shown comes from the backend job (GET /jobs/{id}); downloads come
from GET /jobs/{id}/exports/{key}, which renders the same objects. All
dynamic text is HTML-escaped before it is placed in custom markup.
"""

import html
import os
import re
import time
from datetime import datetime, timezone

import requests
import streamlit as st

# 127.0.0.1 rather than localhost: on Windows "localhost" tries IPv6 first and
# every request waits ~2 s before falling back to the IPv4-only backend.
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
POLL_SECONDS = 1
NOT_SPECIFIED = "Not specified"
FORMATS = ["wav", "mp3", "m4a", "ogg", "flac", "webm", "mp4"]

STAGES = [
    ("validate", "Audio validation"),
    ("normalize", "Audio normalization"),
    ("transcribe", "Speech transcription"),
    ("refine", "Transcript refinement"),
    ("document", "Meeting documentation"),
    ("export", "Export"),
]
STATE_LABEL = {"pending": "Pending", "running": "Processing", "done": "Completed",
               "failed": "Failed", "skipped": "Skipped"}
STATE_ICON = {"pending": "○", "running": "◐", "done": "✓", "failed": "✕", "skipped": "–"}

# Plain-language guidance per backend error code (the reason itself comes from the backend)
ERROR_HINTS = {
    "empty": "Upload a recording that contains audio.",
    "unsupported_format": "Convert the recording to WAV, MP3, M4A, OGG, FLAC, WEBM or MP4.",
    "unreadable": "The file may be damaged. Try exporting the recording again.",
    "decode_failed": "The file may be damaged. Try exporting the recording again.",
    "no_audio_stream": "Upload a file that contains an audio track.",
    "too_short": "Upload a longer recording.",
    "too_large": f"Upload a recording smaller than {MAX_UPLOAD_MB} MB.",
    "silent": "Check that the recording actually contains speech.",
    "no_speech": "Upload a recording containing clear spoken English.",
    "missing_api_key": "The service is not fully configured. Please contact the administrator.",
    "auth_failed": "The service is not fully configured. Please contact the administrator.",
    "rate_limited": "The AI service is busy. Wait a minute and try again.",
    "provider_error": "The AI service is temporarily unavailable. Try again shortly.",
    "timeout": "The AI service took too long to respond. Try again.",
    "network_error": "The server could not reach the AI service. Try again shortly.",
    "invalid_response": "The AI model returned unusable output. Try again.",
    "refused": "The AI model declined to process this recording.",
}
KIND_LABEL = {"summary": "Summary", "minute": "Minute", "decision": "Decision",
              "proposal": "Proposal", "action_item": "Action item"}

# Downloadable deliverables (each as PDF and Word) and machine-readable data files
DELIVERABLES = [
    ("raw_transcript", "Raw transcript", "Speech-to-text result before language-model refinement."),
    ("refined_transcript", "Refined transcript", "After domain-aware terminology correction, with every correction listed."),
    ("minutes", "Meeting minutes", "Concise summary and organized account of the main discussion points."),
    ("decisions", "Key decisions", "Decisions reached, with evidence. Proposals are listed separately."),
    ("action_items", "Action items", "Tasks with description and any stated owner or deadline."),
    ("meeting_record", "Complete meeting record", "All of the above in one document, with both transcripts."),
]
DATA_FILES = [
    ("meeting_record_json", "Complete record", "JSON"),
    ("decisions_json", "Decisions", "JSON"),
    ("action_items_json", "Action items", "JSON"),
    ("action_items", "Action items", "CSV"),
    ("raw_transcript", "Raw transcript", "TXT"),
    ("refined_transcript", "Refined transcript", "TXT"),
    ("minutes", "Meeting minutes", "Markdown"),
    ("meeting_record_md", "Complete record", "Markdown"),
]

# Logo: a person in a chair, wearing headphones, at a podcast microphone
LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none" stroke-linecap="round" '
    'stroke-linejoin="round"><g transform="translate(2.5 0.5)">'
    '<path d="M14.5 21.5 Q13.5 31 16.5 40.5 H31" stroke="#60A5FA" stroke-width="2.6"/>'
    '<path d="M23.5 41 V50 M17 51.5 H30" stroke="#60A5FA" stroke-width="2.6"/>'
    '<circle cx="17" cy="54.2" r="1.6" fill="#60A5FA"/><circle cx="30" cy="54.2" r="1.6" fill="#60A5FA"/>'
    '<path d="M23.8 24.5 L20.8 37.5 H31.5 V49.5 H35.5" stroke="#FFFFFF" stroke-width="4.4"/>'
    '<path d="M24.2 26.5 L29.5 32.5 L36.5 30.4" stroke="#FFFFFF" stroke-width="3.6"/>'
    '<circle cx="26" cy="15.4" r="5.6" fill="#FFFFFF"/>'
    '<path d="M21.4 14.2 A7.1 7.1 0 0 1 32.4 11.6" stroke="#22D3EE" stroke-width="2.4"/>'
    '<rect x="19.2" y="13.2" width="3.8" height="6.2" rx="1.8" fill="#22D3EE"/>'
    '<path d="M34.5 31.5 H53" stroke="#60A5FA" stroke-width="2.6"/>'
    '<rect x="38.6" y="11" width="7.2" height="12.6" rx="3.6" stroke="#22D3EE" stroke-width="2.4"/>'
    '<path d="M39.6 16 H44.8 M39.6 19 H44.8" stroke="#22D3EE" stroke-width="1.4"/>'
    '<path d="M36.4 19.5 A5.8 5.8 0 0 0 48 19.5" stroke="#22D3EE" stroke-width="2"/>'
    '<path d="M42.2 25.4 V31" stroke="#22D3EE" stroke-width="2.4"/></g></svg>'
)
# Browser-tab icon: the same drawing on a rounded navy tile
FAVICON_SVG = LOGO_SVG.replace(
    '<g transform="translate(2.5 0.5)">',
    '<rect width="64" height="64" rx="14" fill="#0E2350"/><g transform="translate(2.5 0.5)">', 1)


# Small stroke icons (inline SVG, no external assets)
def _svg(path: str, size: int = 22) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{path}</svg>')


ICON = {
    "file": _svg('<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>'
                 '<path d="M9 17v-4M12 18v-6M15 17v-4"/>', 26),
    "doc": _svg('<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>'
                '<path d="M8 13h8M8 17h5"/>', 20),
}

st.set_page_config(page_title="Scripted", page_icon=FAVICON_SVG, layout="wide")

CSS = """
<style>
:root {
  --bg:#081A38; --panel:#0E2350; --border:rgba(96,140,220,0.22); --border-strong:rgba(96,165,250,0.45);
  --text:#E6EDF7; --muted:#9FB1CF; --dim:#6F82A3;
  --blue:#3B82F6; --blue-2:#60A5FA; --cyan:#22D3EE; --violet:#A78BFA; --teal:#2DD4BF;
  --ok:#34D399; --ok-bg:rgba(52,211,153,0.10); --warn:#FBBF24; --warn-bg:rgba(251,191,36,0.10);
  --bad:#F87171; --bad-bg:rgba(248,113,113,0.10);
}
.stApp {
  background:
    radial-gradient(1100px 520px at 85% -10%, rgba(59,130,246,0.16), transparent 60%),
    radial-gradient(900px 480px at -10% 30%, rgba(34,211,238,0.07), transparent 60%),
    linear-gradient(180deg, #081A38 0%, #06142F 100%);
}
.block-container { max-width: 1180px; padding-top: 1.4rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.01em; }
p, li { color: var(--text); }

/* Hero */
.logo-tile { border-radius:12px; display:flex; align-items:center; justify-content:center; flex:0 0 auto;
             background:linear-gradient(145deg,#12306B,#0A1D44); border:1px solid rgba(96,165,250,0.55);
             box-shadow:0 0 18px rgba(59,130,246,0.40); }
.logo-tile svg { width:84%; height:84%; }
.hero-top { display:flex; align-items:center; gap:1rem; }
.hero .htitle { font-size:2.5rem; font-weight:800; margin:0; color:#F4F7FC; line-height:1.05; letter-spacing:-0.02em; }
.hero .htitle .hl { color:var(--blue-2); text-shadow:0 0 24px rgba(96,165,250,0.35); }
.hero .kicker { font-size:0.74rem; font-weight:700; letter-spacing:0.14em; color:var(--blue-2); text-transform:uppercase; }
.hero .tag { color:var(--muted); font-size:1.05rem; margin:0.9rem 0 1.4rem 0; }

/* Cards */
.card { background:linear-gradient(180deg, rgba(17,43,94,0.72), rgba(14,35,80,0.72)); border:1px solid var(--border);
        border-radius:16px; padding:1.05rem 1.2rem; margin-bottom:0.8rem; box-shadow:0 8px 24px rgba(0,0,0,0.18); }
.card.glow { border-color:var(--border-strong); box-shadow:0 0 0 1px rgba(59,130,246,0.08), 0 10px 30px rgba(37,99,235,0.18); }
.card .title { font-size:1.04rem; font-weight:650; color:#F4F7FC; margin:0.2rem 0 0.45rem 0; line-height:1.4; }
.card p { margin:0.2rem 0; color:var(--text); line-height:1.6; }
div[data-testid="stVerticalBlockBorderWrapper"] {
  background:linear-gradient(180deg, rgba(17,43,94,0.55), rgba(12,31,72,0.55));
  border-color:var(--border) !important; border-radius:16px !important;
}
.eyebrow { font-size:0.68rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--blue-2);
           margin-bottom:0.35rem; }
.muted { color:var(--muted); font-size:0.88rem; line-height:1.5; }
.quote { border-left:3px solid var(--blue); background:rgba(59,130,246,0.07); padding:0.5rem 0.8rem; margin:0.5rem 0;
         color:#D7E3F5; font-size:0.92rem; border-radius:0 8px 8px 0; line-height:1.5; }
.chip { display:inline-block; font-size:0.68rem; font-weight:700; letter-spacing:0.06em; padding:0.16rem 0.55rem;
        border-radius:999px; border:1px solid var(--border); color:var(--muted); margin-right:0.35rem; }
.chip.blue { color:var(--blue-2); border-color:rgba(96,165,250,0.4); background:rgba(59,130,246,0.12); }
.chip.ok { color:var(--ok); border-color:rgba(52,211,153,0.35); background:var(--ok-bg); }
.chip.warn { color:var(--warn); border-color:rgba(251,191,36,0.35); background:var(--warn-bg); }

/* Upload */
.step-head h3 { margin:0 0 0.25rem 0; font-size:1.45rem; color:#F4F7FC; }
div[data-testid="stFileUploaderDropzone"] {
  background:rgba(8,26,56,0.55); border:1.5px dashed rgba(96,165,250,0.45); border-radius:14px;
  padding:1.6rem 1rem 1.4rem 1rem; flex-direction:column; justify-content:center; text-align:center; gap:0.6rem;
}
div[data-testid="stFileUploaderDropzone"]::before {
  content:""; width:46px; height:46px; display:block; margin:0 auto; border-radius:50%;
  background: rgba(59,130,246,0.14) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2360A5FA' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M7 18a4.5 4.5 0 0 1-.6-8.96A6 6 0 0 1 18 8.5a4 4 0 0 1 .5 7.97'/><path d='M12 12v8M9 15l3-3 3 3'/></svg>") center/24px no-repeat;
  box-shadow:0 0 20px rgba(59,130,246,0.35);
}
div[data-testid="stFileUploaderDropzoneInstructions"] { justify-content:center; }
div[data-testid="stFileUploaderDropzone"] button { border-color:rgba(96,165,250,0.5); background:rgba(59,130,246,0.12); }
.drop-hint { text-align:center; color:var(--text); font-weight:600; margin:0.2rem 0 -0.2rem 0; font-size:0.95rem; }
.file { display:flex; align-items:center; gap:0.85rem; }
.file .ic { width:46px; height:46px; border-radius:12px; display:flex; align-items:center; justify-content:center;
            color:var(--blue-2); background:rgba(59,130,246,0.14); box-shadow:0 0 16px rgba(59,130,246,0.30); }
.file .name { font-weight:700; color:#F4F7FC; }

/* Buttons */
.stButton button[kind="primary"] {
  background:linear-gradient(180deg, #3B82F6, #2563EB); border:1px solid #60A5FA; color:#fff; border-radius:10px;
  font-weight:600; box-shadow:0 0 18px rgba(59,130,246,0.45);
}
.stButton button[kind="primary"]:hover { background:linear-gradient(180deg, #60A5FA, #3B82F6); }
div[data-testid="stDownloadButton"] button { width:100%; border-radius:10px; font-weight:600;
  background:rgba(59,130,246,0.10); border:1px solid rgba(96,165,250,0.45); color:#DCE8FA; }
div[data-testid="stDownloadButton"] button:hover { background:rgba(59,130,246,0.22); border-color:#60A5FA; }

/* Processing */
.proc-head { display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; margin-bottom:0.3rem; }
.proc-head .title { font-size:1.08rem; font-weight:700; color:#F4F7FC; }
.clocks { display:flex; gap:1.8rem; align-items:flex-end; flex:0 0 auto; }
.clock { font-variant-numeric:tabular-nums; font-size:1.7rem; font-weight:800; color:var(--blue-2);
         text-shadow:0 0 18px rgba(96,165,250,0.35); line-height:1; text-align:right; white-space:nowrap; }
.clock.eta { color:var(--cyan); text-shadow:0 0 18px rgba(34,211,238,0.30); }
.clock.wait { font-size:1.05rem; color:var(--muted); text-shadow:none; }
.clock-label { font-size:0.66rem; font-weight:700; letter-spacing:0.12em; color:var(--muted); text-transform:uppercase;
               text-align:right; margin-bottom:0.2rem; }
.now { font-size:0.88rem; color:var(--muted); margin:0.2rem 0 0.7rem 0; }
.now b { color:var(--text); }
div[data-testid="stProgress"] > div > div > div > div { background:linear-gradient(90deg, #2563EB, #22D3EE) !important; }
div[data-testid="stSpinner"] { color:var(--blue-2); }
.stages { display:grid; grid-template-columns:repeat(6, 1fr); gap:0.55rem; }
.stage { background:rgba(14,35,80,0.65); border:1px solid var(--border); border-radius:14px; padding:0.7rem 0.75rem; }
.stage .num { font-size:0.68rem; color:var(--dim); font-weight:700; letter-spacing:0.08em; display:flex; justify-content:space-between; }
.stage .num .dur { color:var(--muted); font-variant-numeric:tabular-nums; letter-spacing:0; }
.stage .name { font-size:0.85rem; font-weight:650; color:var(--text); margin:0.15rem 0 0.35rem 0; line-height:1.2; }
.stage .state { font-size:0.78rem; font-weight:700; }
.stage .msg { font-size:0.7rem; color:var(--muted); margin-top:0.25rem; line-height:1.3; }
.stage.pending .state, .stage.skipped .state { color:var(--dim); }
.stage.done .state { color:var(--ok); }
.stage.done { border-color:rgba(52,211,153,0.28); }
.stage.running { border-color:var(--blue-2); background:rgba(59,130,246,0.14); box-shadow:0 0 22px rgba(59,130,246,0.35); }
.stage.running .state { color:var(--blue-2); }
.stage.failed { border-color:rgba(248,113,113,0.6); background:var(--bad-bg); }
.stage.failed .state { color:var(--bad); }

/* Banners and metrics */
.banner { border-radius:16px; padding:1rem 1.2rem; margin:0.9rem 0; border:1px solid; }
.banner.ok { background:linear-gradient(90deg, rgba(52,211,153,0.12), rgba(59,130,246,0.06)); border-color:rgba(52,211,153,0.35); }
.banner.bad { background:var(--bad-bg); border-color:rgba(248,113,113,0.45); }
.banner .head { font-weight:800; font-size:1.08rem; color:#F4F7FC; }
.banner.ok .head { color:#B9F3DC; }
.banner.bad .head { color:#FFC9C9; letter-spacing:0.04em; }
.banner .k { font-size:0.68rem; font-weight:700; letter-spacing:0.12em; color:var(--muted); text-transform:uppercase; margin-top:0.6rem; }
.metrics { display:grid; grid-template-columns:repeat(5, 1fr); gap:0.6rem; margin:0.2rem 0 1rem 0; }
.metric { background:linear-gradient(180deg, rgba(17,43,94,0.8), rgba(14,35,80,0.8)); border:1px solid var(--border);
          border-radius:14px; padding:0.8rem 1rem; }
.metric .label { font-size:0.68rem; color:var(--muted); font-weight:700; letter-spacing:0.1em; text-transform:uppercase; }
.metric .value { font-size:1.9rem; font-weight:800; color:#F4F7FC; line-height:1.25; }
.metric.accent .value { color:var(--blue-2); text-shadow:0 0 18px rgba(96,165,250,0.35); }

div[data-testid="stButtonGroup"] button { border-radius:10px !important; font-weight:600; }

/* Transcript */
.line { display:flex; gap:0.8rem; padding:0.42rem 0.2rem; border-bottom:1px solid rgba(96,140,220,0.12); font-size:0.92rem; }
.line .sid { color:var(--dim); font-size:0.72rem; font-weight:700; min-width:1.8rem; padding-top:0.15rem; }
.line .ts { color:var(--blue-2); font-variant-numeric:tabular-nums; white-space:nowrap; font-size:0.78rem; padding-top:0.1rem; }
.line .tx { color:var(--text); }
.seg-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.55rem; }
.seg-head .id { font-size:0.72rem; font-weight:700; letter-spacing:0.1em; color:var(--muted); }
.seg-head .id b { color:var(--blue-2); }
.cmp { display:grid; grid-template-columns:1fr 1fr; gap:0.7rem; }
.cmp .box { border-radius:12px; padding:0.65rem 0.8rem; font-size:0.92rem; line-height:1.5; }
.cmp .raw { background:rgba(6,20,47,0.6); border:1px solid rgba(159,177,207,0.22); color:#C9D4E6; }
.cmp .ref { background:rgba(59,130,246,0.09); border:1px solid rgba(96,165,250,0.4); color:#EAF2FF; }
.cmp .lbl { font-size:0.66rem; font-weight:800; letter-spacing:0.12em; margin-bottom:0.25rem; }
.cmp .raw .lbl { color:var(--muted); }
.cmp .ref .lbl { color:var(--blue-2); }

/* Fields and stats */
.fields { display:grid; grid-template-columns:2fr 1fr 1fr; gap:0.9rem; }
.field .k { font-size:0.66rem; font-weight:700; letter-spacing:0.12em; color:var(--muted); text-transform:uppercase; }
.field .v { font-size:0.97rem; color:#F4F7FC; font-weight:600; margin-top:0.1rem; }
.field .v.none { color:var(--dim); font-style:italic; font-weight:500; }
.stat { display:flex; justify-content:space-between; padding:0.42rem 0; border-bottom:1px solid rgba(96,140,220,0.12);
        font-size:0.9rem; color:var(--muted); }
.stat b { color:#F4F7FC; }
.section-h { font-size:1.25rem; font-weight:750; color:#F4F7FC; margin:0.2rem 0 0.15rem 0; }

/* Downloads */
.dl-head { display:flex; gap:0.7rem; align-items:flex-start; margin-bottom:0.55rem; }
.dl-head .ic { width:38px; height:38px; flex:0 0 38px; border-radius:10px; display:flex; align-items:center;
               justify-content:center; color:var(--blue-2); background:rgba(59,130,246,0.14); }
.dl-head .t { font-weight:700; color:#F4F7FC; font-size:0.98rem; }
.dl-head .d { font-size:0.8rem; color:var(--muted); line-height:1.35; }

@media (max-width: 980px) {
  .stages { grid-template-columns:repeat(3, 1fr); }
  .hero .htitle { font-size:2.1rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def ts(seconds: float) -> str:
    total = int(seconds)
    h, m, s = total // 3600, total % 3600 // 60, total % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def duration(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 60 else ts(seconds)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def seg_ids(ids: list[int]) -> str:
    return ", ".join(f"{i:02d}" for i in ids)


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def card(body: str, glow: bool = False) -> None:
    md(f'<div class="card{" glow" if glow else ""}">{body}</div>')


def eyebrow(text: str) -> str:
    return f'<div class="eyebrow">{esc(text)}</div>'


def logo(size: int) -> str:
    return f'<div class="logo-tile" style="width:{size}px;height:{size}px">{LOGO_SVG}</div>'


def get(path: str, timeout: float = 10) -> requests.Response:
    return requests.get(f"{BACKEND_URL}{path}", timeout=timeout)


def api_error(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    return detail if isinstance(detail, str) else "The server could not process the request."


@st.cache_data(ttl=30, show_spinner=False)
def backend_online() -> bool:
    try:
        return get("/health", timeout=3).ok
    except requests.RequestException:
        return False


def finished_job(job_id: str) -> dict | None:
    """A completed job never changes, so it is fetched once per session."""
    return st.session_state.get("finished_jobs", {}).get(job_id)


def load_exports(job_id: str) -> dict[str, tuple[dict, bytes]]:
    """Fetch the export files of a finished job once and keep them for this session."""
    cache = st.session_state.setdefault("exports", {})
    if job_id not in cache:
        files: dict[str, tuple[dict, bytes]] = {}
        try:
            for f in get(f"/jobs/{job_id}/exports", timeout=60).json():
                resp = get(f"/jobs/{job_id}/exports/{f['key']}", timeout=60)
                if resp.ok:
                    files[f["key"]] = (f, resp.content)
        except (requests.RequestException, ValueError):
            return {}
        cache[job_id] = files
    return cache[job_id]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

md(
    f'<div class="hero"><div class="hero-top">{logo(64)}<div><div class="kicker">AI meeting assistant</div>'
    '<div class="htitle">Script<span class="hl">ed</span></div></div></div>'
    '<div class="tag">From recorded meetings to grounded, actionable documentation.</div></div>'
)

if not backend_online():
    md('<div class="banner bad"><div class="head">Service unavailable</div>'
       '<div class="muted">The processing service is not reachable right now. Please try again shortly.</div></div>')

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

job_id = st.session_state.get("job_id")
with st.container(border=True):
    md('<div class="step-head"><div class="eyebrow">Step 1</div><h3>Upload your meeting</h3>'
       '<div class="muted">Upload a recorded English meeting to generate transcripts, decisions and action items. '
       f'Supported: WAV, MP3, M4A, OGG, FLAC, WEBM, MP4 · Max {MAX_UPLOAD_MB} MB. '
       'Audio is deleted after processing.</div></div>')
    md('<div class="drop-hint" style="margin-top:0.8rem">Drag and drop your meeting file here</div>')
    uploaded = st.file_uploader("Meeting recording", type=FORMATS, label_visibility="collapsed")
    if uploaded is not None:
        ext = uploaded.name.rsplit(".", 1)[-1].upper() if "." in uploaded.name else "FILE"
        kind = "Video recording" if ext in ("MP4", "WEBM") else "Audio recording"
        left, right = st.columns([3, 1], vertical_alignment="center")
        with left:
            md(f'<div class="file"><div class="ic">{ICON["file"]}</div><div>'
               f'<div class="name">{esc(uploaded.name)}</div>'
               f'<div class="muted">{kind} · {esc(ext)} · {uploaded.size / 1_048_576:.2f} MB</div></div></div>')
        with right:
            process = st.button("Process Meeting", type="primary", use_container_width=True)
        if process:
            with st.spinner("Uploading recording…"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/jobs",
                        files={"file": (uploaded.name, uploaded.getvalue(),
                                        uploaded.type or "application/octet-stream")},
                        timeout=300,
                    )
                except requests.RequestException:
                    resp = None
            if resp is None:
                st.session_state.pop("job_id", None)
                st.session_state["upload_error"] = "The processing service could not be reached."
                job_id = None
            elif resp.ok:
                st.session_state["job_id"] = job_id = resp.json()["id"]
                st.session_state.pop("upload_error", None)
                st.session_state["nav"] = "Overview"
            else:
                st.session_state.pop("job_id", None)
                st.session_state["upload_error"] = api_error(resp)
                job_id = None

if st.session_state.get("upload_error"):
    md(f'<div class="banner bad"><div class="head">UPLOAD REJECTED</div>'
       f'<div>{esc(st.session_state["upload_error"])}</div></div>')

if not job_id:
    st.stop()


# ---------------------------------------------------------------------------
# Processing: real backend stage status, live timer, rough time left, progress
# ---------------------------------------------------------------------------

# Rough cost of each stage, fitted to real runs of this pipeline, as
# (fixed seconds, seconds per second of audio, seconds per transcript word).
# Transcription is mostly the audio upload, so it scales with recording length;
# the two language-model stages and the export scale with the word count.
STAGE_COST = {
    "validate": (1.0, 0.0, 0.0),
    "normalize": (1.0, 0.002, 0.0),
    "transcribe": (5.0, 0.24, 0.0),
    "refine": (20.0, 0.0, 0.03),
    "document": (12.0, 0.0, 0.006),
    "export": (2.0, 0.0, 0.0015),
}
WORDS_PER_SECOND = 2.6  # typical speech rate, used until the transcript exists
# The validation and normalization stage messages start with the recording length, e.g. "180.0s, mp3 / mp3"
AUDIO_LENGTH = re.compile(r"^(\d+(?:\.\d+)?)s\b")


def stage_seconds(stage: dict, now: datetime) -> float | None:
    start, end = parse_time(stage.get("started_at")), parse_time(stage.get("finished_at"))
    if start is None:
        return None
    return max(0.0, ((end or now) - start).total_seconds())


def job_seconds(job: dict, now: datetime) -> float:
    start = parse_time(job.get("created_at"))
    if start is None:
        return 0.0
    ends = [parse_time(s.get("finished_at")) for s in job["stages"]]
    end = max((e for e in ends if e), default=None) if job["status"] in ("done", "failed") else now
    return max(0.0, ((end or now) - start).total_seconds())


def audio_length(job: dict) -> float | None:
    for stage in job["stages"]:
        match = AUDIO_LENGTH.match(stage.get("message") or "")
        if stage["name"] in ("validate", "normalize") and match:
            return float(match.group(1))
    return None


def time_left(job: dict, now: datetime) -> tuple[float, bool] | None:
    """Rough seconds until the job finishes, and whether the current stage is running long."""
    audio = audio_length(job)
    if audio is None:
        return None
    raw = job.get("raw_transcript")
    words = sum(len(s["text"].split()) for s in raw["segments"]) if raw else audio * WORDS_PER_SECOND
    left, slow = 0.0, False
    for stage in job["stages"]:
        fixed, per_second, per_word = STAGE_COST.get(stage["name"], (5.0, 0.0, 0.0))
        cost = fixed + per_second * audio + per_word * words
        if stage["status"] == "pending":
            left += cost
        elif stage["status"] == "running":
            spent = stage_seconds(stage, now) or 0.0
            # A stage that overruns its estimate still has a little left, never zero
            left += max(cost - spent, 0.15 * cost, 2.0)
            slow = spent > 1.5 * cost + 10
    return left, slow


def rough(seconds: float) -> str:
    if seconds < 20:
        return "Almost done"
    if seconds < 50:
        return "< 1 min"
    if seconds < 90:
        return "~1 min"
    if seconds < 90 * 60:
        return f"~{int(seconds / 60 + 0.5)} min"
    return f"~{seconds / 3600:.1f} h"


def stages_html(job: dict, now: datetime) -> str:
    by_name = {s["name"]: s for s in job["stages"]}
    cells = []
    for n, (key, label) in enumerate(STAGES, 1):
        stage = by_name.get(key, {"status": "pending", "message": None})
        state = stage["status"]
        secs = stage_seconds(stage, now)
        dur = f'<span class="dur">{duration(secs)}</span>' if secs is not None and state in ("done", "running", "failed") else ""
        msg = stage.get("message") if state in ("done", "running") else ""
        cells.append(
            f'<div class="stage {state}"><div class="num"><span>{n:02d}</span>{dur}</div><div class="name">{label}</div>'
            f'<div class="state">{STATE_ICON[state]} {STATE_LABEL[state]}</div>'
            f'{f"<div class=msg>{esc(msg)}</div>" if msg else ""}</div>'
        )
    return '<div class="stages">' + "".join(cells) + "</div>"


def render_progress(job: dict, header, bar, grid) -> None:
    now = datetime.now(timezone.utc)
    done = sum(s["status"] == "done" for s in job["stages"])
    running = next((s for s in job["stages"] if s["status"] == "running"), None)
    labels = dict(STAGES)
    elapsed = ts(job_seconds(job, now))
    if job["status"] == "done":
        title, clocks = "Processing complete", [("Total time", elapsed, "")]
        now_line = f"All {len(STAGES)} stages completed."
    elif job["status"] == "failed":
        title, clocks = "Processing stopped", [("Time", elapsed, "")]
        now_line = "A stage failed. See the details below."
    else:
        estimate = time_left(job, now)
        title = "Processing your meeting…"
        clocks = [("Elapsed", elapsed, ""),
                  ("Est. time left", rough(estimate[0]), " eta") if estimate else
                  ("Est. time left", "Estimating…", " eta wait")]
        if running:
            now_line = (f"Now: <b>{esc(labels.get(running['name'], running['name']))}</b> · running for "
                        f"{duration(stage_seconds(running, now) or 0)}")
            if estimate and estimate[1]:
                now_line += " · taking longer than usual"
        else:
            now_line = "Waiting to start…"
    header.markdown(
        f'<div class="proc-head"><div><div class="eyebrow">Step 2 · Processing</div>'
        f'<div class="title">{esc(title)}</div><div class="muted">{esc(job["filename"])}</div></div>'
        '<div class="clocks">' + "".join(
            f'<div><div class="clock-label">{label}</div><div class="clock{cls}">{esc(value)}</div></div>'
            for label, value, cls in clocks) + '</div></div>'
        f'<div class="now">{now_line}</div>', unsafe_allow_html=True)
    bar.progress(done / len(STAGES), text=f"{done} of {len(STAGES)} stages completed")
    grid.markdown(stages_html(job, now), unsafe_allow_html=True)


job = finished_job(job_id)
with st.container(border=True):
    header, bar, grid = st.empty(), st.empty(), st.empty()
    if job is None:
        with st.spinner("Processing — results appear here automatically when ready…"):
            while True:
                try:
                    resp = get(f"/jobs/{job_id}")
                except requests.RequestException:
                    md('<div class="banner bad"><div class="head">CONNECTION LOST</div>'
                       '<div>The processing service stopped responding.</div></div>')
                    st.stop()
                if resp.status_code == 404:
                    st.session_state.pop("job_id", None)
                    md('<div class="banner bad"><div class="head">ANALYSIS NOT FOUND</div>'
                       '<div>This analysis is no longer available. Please process the recording again.</div></div>')
                    st.stop()
                if not resp.ok:
                    md('<div class="banner bad"><div class="head">SERVICE ERROR</div>'
                       '<div>The processing service returned an error.</div></div>')
                    st.stop()
                current = resp.json()
                render_progress(current, header, bar, grid)
                if current["status"] in ("done", "failed"):
                    job = current
                    st.session_state.setdefault("finished_jobs", {})[job_id] = job
                    break
                time.sleep(POLL_SECONDS)
    else:
        render_progress(job, header, bar, grid)

raw, refined, record = job.get("raw_transcript"), job.get("refined_transcript"), job.get("record")

if job["status"] == "failed":
    failed = next((label for key, label in STAGES
                   for s in job["stages"] if s["name"] == key and s["status"] == "failed"), "Processing")
    hint = ERROR_HINTS.get(job.get("error_code") or "", "Please try again.")
    md(f'<div class="banner bad"><div class="head">PROCESSING FAILED</div>'
       f'<div style="margin-top:0.2rem">Unable to process this recording. The failure occurred during '
       f'<b>{esc(failed.lower())}</b>.</div>'
       f'<div class="k">Reason</div><div>{esc(job.get("error"))}</div>'
       f'<div class="k">What you can try</div><div>{esc(hint)}</div></div>')
    if not raw:
        st.stop()
else:
    md('<div class="banner ok"><div class="head">Meeting analysis complete</div>'
       '<div class="muted">Your transcript and grounded meeting documentation are ready.</div></div>')

# ---------------------------------------------------------------------------
# Metrics (from the actual job result)
# ---------------------------------------------------------------------------

n_segments = len(raw["segments"]) if raw else 0
n_refined = sum(r["status"] == "refined" for r in refined["refinements"]) if refined else "–"
metrics = [
    ("Segments", n_segments, False),
    ("Refined", n_refined, False),
    ("Decisions", len(record["decisions"]) if record else "–", True),
    ("Action Items", len(record["action_items"]) if record else "–", True),
    ("Proposals", len(record["proposals"]) if record else "–", False),
]
md('<div class="metrics">' + "".join(
    f'<div class="metric{" accent" if accent else ""}"><div class="label">{label}</div>'
    f'<div class="value">{value}</div></div>' for label, value, accent in metrics) + "</div>")

nav = st.segmented_control("Section", ["Overview", "Transcript", "Decisions & Actions", "Downloads"],
                           default="Overview", key="nav", label_visibility="collapsed") or "Overview"
md('<div style="height:0.4rem"></div>')

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

if nav == "Overview":
    left, right = st.columns([2, 1], gap="medium")
    with left:
        if record and record.get("summary"):
            card(eyebrow("Meeting summary") + f"<p>{esc(record['summary'])}</p>", glow=True)
        elif record:
            card(eyebrow("Meeting summary") + '<p class="muted">The generated summary was withheld because it '
                 "could not be fully grounded in the transcript.</p>")
        else:
            card(eyebrow("Meeting summary") + '<p class="muted">No summary was produced.</p>')
        if record and record["minutes"]:
            md(eyebrow("Meeting minutes"))
            for m in record["minutes"]:
                card(f'<div class="title">{esc(m["topic"])}</div><p>{esc(m["discussion"])}</p>'
                     f'<p class="muted">Segments {seg_ids(m["segment_ids"])}</p>')
    with right:
        rows = [("Processing time", ts(job_seconds(job, datetime.now(timezone.utc))))]
        if raw:
            rows += [("Duration", ts(raw.get("duration_seconds") or 0)), ("Transcript segments", n_segments)]
        if refined:
            refs = refined["refinements"]
            rows += [("Segments refined", sum(r["status"] == "refined" for r in refs)),
                     ("Refinements rejected", sum(r["status"] == "fallback" for r in refs))]
        if record:
            rows += [("Minutes topics", len(record["minutes"])), ("Withheld items", len(record["rejected_items"]))]
        card(eyebrow("Meeting statistics") + "".join(
            f'<div class="stat"><span>{label}</span><b>{esc(value)}</b></div>' for label, value in rows))
        models = [m for m in [raw and raw.get("stt_model"), refined and refined.get("refine_model"),
                              record and record.get("document_model")] if m]
        if models:
            card(eyebrow("Models used") + "".join(f'<span class="chip blue">{esc(m)}</span>' for m in models))

# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


def transcript_lines(segments: list[dict]) -> str:
    return "".join(
        f'<div class="line"><span class="sid">{s["id"]:02d}</span>'
        f'<span class="ts">{ts(s["start"])} → {ts(s["end"])}</span><span class="tx">{esc(s["text"])}</span></div>'
        for s in segments if s["text"].strip())


if nav == "Transcript":
    view = st.segmented_control("View", ["Raw vs Refined", "Raw Transcript", "Refined Transcript"],
                                default="Raw vs Refined", key="transcript_view",
                                label_visibility="collapsed") or "Raw vs Refined"
    if view == "Raw Transcript":
        if raw:
            md(f'<div class="section-h">Raw transcript</div><div class="muted" style="margin-bottom:0.6rem">'
               f'Exactly as produced by speech-to-text ({esc(raw["stt_model"])}). Never modified.</div>')
            with st.container(height=540, border=True):
                md(transcript_lines(raw["segments"]))
        else:
            st.info("No raw transcript is available.")
    elif view == "Refined Transcript":
        if refined:
            md(f'<div class="section-h">Refined transcript</div><div class="muted" style="margin-bottom:0.6rem">'
               f'Recognition errors, terminology and formatting corrected by {esc(refined["refine_model"])}. '
               'Same segments and timestamps as the raw transcript.</div>')
            with st.container(height=540, border=True):
                md(transcript_lines(refined["segments"]))
        else:
            st.info("No refined transcript is available.")
    else:
        if refined:
            md('<div class="section-h">Raw vs refined</div><div class="muted" style="margin-bottom:0.4rem">'
               'The raw transcript is never modified. Each refined segment is shown next to its original.</div>')
            only_changed = st.checkbox("Show only segments that changed")
            badge = {"refined": '<span class="chip blue">REFINED</span>',
                     "unchanged": '<span class="chip">UNCHANGED</span>',
                     "fallback": '<span class="chip warn">REFINEMENT REJECTED · RAW KEPT</span>'}
            with st.container(height=620, border=False):
                for r in refined["refinements"]:
                    if only_changed and r["status"] == "unchanged":
                        continue
                    extra = ""
                    if r["status"] == "fallback":
                        extra = (f'<div class="muted" style="margin-top:0.5rem">Rejected edit: '
                                 f'“{esc(r.get("rejected_text") or "")}” — {esc("; ".join(r.get("issues") or []))}</div>')
                    card(f'<div class="seg-head"><span class="id">SEGMENT <b>{r["segment_id"]:02d}</b> · '
                         f'{ts(r["start"])} → {ts(r["end"])}</span>{badge.get(r["status"], "")}</div>'
                         f'<div class="cmp"><div class="box raw"><div class="lbl">RAW</div>{esc(r["original_text"])}</div>'
                         f'<div class="box ref"><div class="lbl">REFINED</div>{esc(r["refined_text"])}</div></div>{extra}')
        else:
            st.info("No refined transcript is available.")

# ---------------------------------------------------------------------------
# Decisions & actions
# ---------------------------------------------------------------------------


def evidence(quote: str, ids: list[int]) -> str:
    return (f'<div class="eyebrow" style="margin-top:0.6rem;color:var(--muted)">Evidence</div>'
            f'<div class="quote">“{esc(quote)}”</div><div class="muted">Segments {seg_ids(ids)}</div>')


def field(label: str, value) -> str:
    cls = "v" if value else "v none"
    return f'<div class="field"><div class="k">{label}</div><div class="{cls}">{esc(value or NOT_SPECIFIED)}</div></div>'


if nav == "Decisions & Actions":
    if not record:
        st.info("The meeting record was not generated for this recording.")
    else:
        section = st.segmented_control(
            "Items", ["Decisions", "Action Items", "Proposals / Not Agreed", "Safety Checks"],
            default="Decisions", key="items_view", label_visibility="collapsed") or "Decisions"
        if section == "Decisions":
            md('<div class="section-h">Decisions</div><div class="muted" style="margin-bottom:0.7rem">'
               'Only items the meeting explicitly agreed, confirmed or approved.</div>')
            for n, d in enumerate(record["decisions"], 1):
                card(f'{eyebrow(f"Decision {n:02d}")}<div class="title">{esc(d["decision"])}</div>'
                     + evidence(d["evidence_quote"], d["evidence_segment_ids"]), glow=True)
            if not record["decisions"]:
                card('<p class="muted">No decisions were explicitly agreed in this meeting.</p>')

        elif section == "Action Items":
            md('<div class="section-h">Action items</div><div class="muted" style="margin-bottom:0.7rem">'
               'Only explicit assignments or commitments. Owner and deadline appear only when the recording '
               'states them.</div>')
            for n, a in enumerate(record["action_items"], 1):
                card(f'{eyebrow(f"Action item {n:02d}")}<div class="fields">'
                     f'{field("Task", a["task"])}{field("Owner", a["owner"])}{field("Deadline", a["deadline"])}</div>'
                     + evidence(a["evidence_quote"], a["evidence_segment_ids"]), glow=True)
            if not record["action_items"]:
                card('<p class="muted">No tasks were explicitly assigned in this meeting.</p>')

        elif section == "Proposals / Not Agreed":
            md('<div class="section-h">Proposals / not agreed</div><div class="muted" style="margin-bottom:0.7rem">'
               'Discussed during the meeting but not confirmed as decisions.</div>')
            for n, p in enumerate(record["proposals"], 1):
                card(f'{eyebrow(f"Proposal {n:02d}")}<span class="chip warn">NOT AGREED</span>'
                     f'<div class="title">{esc(p["proposal"])}</div>'
                     + evidence(p["evidence_quote"], p["evidence_segment_ids"]))
            if not record["proposals"]:
                card('<p class="muted">No open proposals.</p>')

        else:
            md('<div class="section-h">Safety checks / withheld items</div><div class="muted" style="margin-bottom:0.7rem">'
               'Items that could not be sufficiently grounded in the transcript are withheld rather than '
               'presented as facts.</div>')
            withheld = record["rejected_items"]
            if not withheld:
                md('<div class="banner ok"><div class="head">✓ No unsupported items were accepted.</div>'
                   '<div class="muted">Every generated item passed the grounding checks; nothing had to be withheld.</div></div>')
            for r in withheld:
                c = r["content"]
                attempted = (c.get("decision") or c.get("proposal") or c.get("task")
                             or c.get("discussion") or c.get("summary") or "")
                reasons = "".join(f"<li>{esc(x)}</li>" for x in r["reasons"])
                details = ""
                if "owner" in c or "deadline" in c:
                    details = (f'<div class="muted">Attempted owner: {esc(c.get("owner") or NOT_SPECIFIED)} · '
                               f'Attempted deadline: {esc(c.get("deadline") or NOT_SPECIFIED)}</div>')
                ev = (evidence(c["evidence_quote"], c.get("evidence_segment_ids") or [])
                      if c.get("evidence_quote") else "")
                card(f'<span class="chip warn">WITHHELD · {esc(KIND_LABEL.get(r["kind"], r["kind"]).upper())}</span>'
                     f'<div class="title">{esc(attempted)}</div>{details}'
                     f'<div class="eyebrow" style="margin-top:0.6rem;color:var(--warn)">Why it was withheld</div>'
                     f'<ul style="margin:0 0 0.3rem 1.1rem">{reasons}</ul>{ev}')

# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

if nav == "Downloads":
    md('<div class="section-h">Export meeting results</div><div class="muted" style="margin-bottom:0.9rem">'
       'Download every deliverable as a PDF or Word document, plus machine-readable data.</div>')
    with st.spinner("Preparing your documents…"):
        available = load_exports(job_id)
    if not available:
        st.info("Nothing is available to download.")

    present = [d for d in DELIVERABLES if f"{d[0]}_pdf" in available or f"{d[0]}_docx" in available]
    if present:
        md(eyebrow("Deliverables · PDF and Word"))
        cols = st.columns(2, gap="small")
        for i, (key, title, desc) in enumerate(present):
            with cols[i % 2], st.container(border=True):
                md(f'<div class="dl-head"><div class="ic">{ICON["doc"]}</div><div><div class="t">{esc(title)}</div>'
                   f'<div class="d">{esc(desc)}</div></div></div>')
                pdf_col, doc_col = st.columns(2, gap="small")
                for col, fmt, label in ((pdf_col, "pdf", "PDF"), (doc_col, "docx", "Word")):
                    if f"{key}_{fmt}" in available:
                        meta, data = available[f"{key}_{fmt}"]
                        with col:
                            st.download_button(f"Download {label}", data=data, file_name=meta["filename"],
                                               mime=meta["mime"], key=f"dl-{key}-{fmt}", on_click="ignore")

    data_files = [d for d in DATA_FILES if d[0] in available]
    if data_files:
        md('<div style="height:0.6rem"></div>' + eyebrow("Machine-readable data"))
        cols = st.columns(4, gap="small")
        for i, (key, title, fmt) in enumerate(data_files):
            meta, data = available[key]
            with cols[i % 4]:
                st.download_button(f"{title} · {fmt}", data=data, file_name=meta["filename"],
                                   mime=meta["mime"], key=f"dl-data-{key}", on_click="ignore")
