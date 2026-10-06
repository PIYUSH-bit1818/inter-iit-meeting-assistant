"""Streamlit UI for the meeting assistant.

Run from the repo root (with the backend running): streamlit run frontend/app.py

Everything shown here comes from the backend job (GET /jobs/{id}); downloads
come from GET /jobs/{id}/exports/{key}, which renders the same objects.
"""

import os
import time

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
POLL_SECONDS = 2
NOT_SPECIFIED = "Not specified"
STAGE_LABELS = {
    "validate": "Audio validation",
    "normalize": "Audio normalization (16 kHz mono)",
    "transcribe": "Speech transcription (Groq Whisper)",
    "refine": "Transcript refinement (Gemini LLM #1)",
    "document": "Meeting documentation (Gemini LLM #2)",
    "export": "Export generation",
}
STATUS_ICONS = {"pending": "⏳", "running": "🔄", "done": "✅", "failed": "❌", "skipped": "⏭️"}
# Extra guidance for some failure codes (the message itself comes from the backend)
ERROR_HINTS = {
    "missing_api_key": "Add the missing key to the .env file and restart the backend.",
    "auth_failed": "Check that the API key in .env is valid.",
    "rate_limited": "The provider's free-tier limit was reached. Wait a minute and try again.",
    "provider_error": "The provider is temporarily unavailable. Try again shortly.",
    "timeout": "The provider took too long to respond. Try again.",
    "network_error": "Check the server's internet connection.",
    "invalid_response": "The model returned unusable output twice. Try again.",
    "unsupported_format": "Convert the recording to WAV, MP3, M4A, OGG, FLAC, WEBM or MP4.",
    "silent": "Check that the recording actually contains speech.",
    "no_speech": "Check that the recording contains spoken English.",
    "too_short": "Upload a longer recording.",
}
KIND_LABELS = {"summary": "Summary", "minute": "Minute", "decision": "Decision",
               "proposal": "Proposal", "action_item": "Action item"}

st.set_page_config(page_title="Meeting Assistant", page_icon="📝", layout="wide")


def ts(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def ids(segment_ids: list[int]) -> str:
    return ", ".join(f"#{i}" for i in segment_ids)


def api_error(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    return detail if isinstance(detail, str) else f"The backend returned an error (HTTP {resp.status_code})."


def get(path: str, timeout: float = 10) -> requests.Response:
    return requests.get(f"{BACKEND_URL}{path}", timeout=timeout)


# ---------------------------------------------------------------------------
# Header and backend status
# ---------------------------------------------------------------------------

st.title("📝 Meeting Assistant")
st.caption("Upload a recorded English meeting → transcript → domain-refined transcript → "
           "summary, minutes, decisions, proposals and action items, each with its evidence.")

with st.sidebar:
    st.subheader("Backend")
    try:
        health = get("/health", timeout=3).json()
        st.success("Connected")
        for key, ok in health.get("config", {}).items():
            st.write(("✅ " if ok else "⚠️ missing: ") + key)
    except requests.RequestException:
        st.error(f"Backend unreachable at {BACKEND_URL}. Start it with "
                 "`uvicorn app.main:app --port 8000` from the backend folder.")
    st.divider()
    st.caption("Pipeline: Groq Whisper (speech-to-text) → Gemini refinement (LLM #1) → "
               "Gemini documentation (LLM #2). Uploaded audio is deleted after processing.")

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Meeting recording (WAV, MP3, M4A, OGG, FLAC, WEBM or MP4)",
    type=["wav", "mp3", "m4a", "ogg", "flac", "webm", "mp4"],
)
if uploaded is not None:
    st.write(f"Selected file: **{uploaded.name}** ({uploaded.size / 1_048_576:.2f} MB)")

if st.button("Process recording", type="primary", disabled=uploaded is None):
    try:
        resp = requests.post(
            f"{BACKEND_URL}/jobs",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
            timeout=300,
        )
    except requests.RequestException:
        st.error("Could not reach the backend. Is it running?")
    else:
        if resp.ok:
            st.session_state["job_id"] = resp.json()["id"]
        else:
            st.session_state.pop("job_id", None)
            st.error(f"Upload rejected: {api_error(resp)}")

job_id = st.session_state.get("job_id")
if not job_id:
    st.stop()

# ---------------------------------------------------------------------------
# Progress (polls the backend; a stage is only shown done once it really is)
# ---------------------------------------------------------------------------


def render_stages(job: dict) -> None:
    for stage in job["stages"]:
        icon = STATUS_ICONS.get(stage["status"], "")
        label = STAGE_LABELS.get(stage["name"], stage["name"])
        message = f" — {stage['message']}" if stage.get("message") else ""
        st.markdown(f"{icon} **{label}** · _{stage['status']}_{message}")


status_box = st.empty()
job = None
while True:
    try:
        resp = get(f"/jobs/{job_id}")
    except requests.RequestException:
        st.error("Lost contact with the backend.")
        st.stop()
    if resp.status_code == 404:
        st.warning("This job is no longer available (the backend may have restarted). Please process the file again.")
        st.session_state.pop("job_id", None)
        st.stop()
    if not resp.ok:
        st.error(api_error(resp))
        st.stop()
    job = resp.json()
    with status_box.container(border=True):
        st.subheader(f"Processing: {job['filename']}")
        render_stages(job)
    if job["status"] in ("done", "failed"):
        break
    time.sleep(POLL_SECONDS)

if job["status"] == "failed":
    hint = ERROR_HINTS.get(job.get("error_code") or "", "")
    st.error(f"**Processing failed:** {job['error']}" + (f"  \n{hint}" if hint else ""))
else:
    st.success("Processing complete.")

raw, refined, record = job.get("raw_transcript"), job.get("refined_transcript"), job.get("record")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

tabs = st.tabs(["Raw transcript", "Refined transcript", "Raw vs refined", "Summary", "Minutes",
                "Decisions", "Proposals / not agreed", "Action items", "Withheld / rejected", "Downloads"])

with tabs[0]:
    st.subheader("Raw transcript")
    if raw:
        st.caption(f"Exactly as returned by {raw['stt_model']} — never modified.")
        for s in raw["segments"]:
            st.markdown(f"`{ts(s['start'])} – {ts(s['end'])}` **#{s['id']}** {s['text']}")
    else:
        st.info("No raw transcript (processing stopped before speech-to-text finished).")

with tabs[1]:
    st.subheader("Refined transcript")
    if refined:
        st.caption(f"Cleaned up by {refined['refine_model']}: recognition errors, terminology, "
                   "punctuation and number formatting. Same segments and timestamps as the raw transcript.")
        for s in refined["segments"]:
            st.markdown(f"`{ts(s['start'])} – {ts(s['end'])}` **#{s['id']}** {s['text']}")
    else:
        st.info("No refined transcript.")

with tabs[2]:
    st.subheader("Raw vs refined, segment by segment")
    if refined:
        counts = {k: sum(r["status"] == k for r in refined["refinements"]) for k in ("refined", "unchanged", "fallback")}
        st.caption(f"{counts['refined']} refined · {counts['unchanged']} unchanged · "
                   f"{counts['fallback']} refinement(s) rejected by the safety checks (raw text kept)")
        st.dataframe([{
            "#": r["segment_id"],
            "time": f"{ts(r['start'])} – {ts(r['end'])}",
            "raw": r["original_text"],
            "refined": r["refined_text"],
            "status": r["status"],
            "edits": "; ".join(f"{c['original_span']} → {c['refined_span']}" for c in r["changes"]),
            "rejected edit": r.get("rejected_text") or "",
            "why rejected": "; ".join(r.get("issues") or []),
        } for r in refined["refinements"]], use_container_width=True, hide_index=True)
    else:
        st.info("No refined transcript.")

if not record:
    for tab in tabs[3:9]:
        with tab:
            st.info("No meeting record (processing stopped before documentation finished).")
else:
    with tabs[3]:
        st.subheader("Summary")
        st.write(record["summary"] or "_Summary withheld: it failed validation (see Withheld / rejected)._")
        st.caption(f"Generated by {record['document_model']} from the refined transcript.")

    with tabs[4]:
        st.subheader("Meeting minutes")
        for m in record["minutes"]:
            st.markdown(f"**{m['topic']}**  \n{m['discussion']}  \n_Segments: {ids(m['segment_ids'])}_")
        if not record["minutes"]:
            st.info("No minutes.")

    with tabs[5]:
        st.subheader("Decisions")
        st.caption("Only things the meeting explicitly agreed, confirmed or approved.")
        for n, d in enumerate(record["decisions"], 1):
            st.markdown(f"**{n}. {d['decision']}**  \nEvidence ({ids(d['evidence_segment_ids'])}): "
                        f"“{d['evidence_quote']}”")
        if not record["decisions"]:
            st.info("No decisions were recorded.")

    with tabs[6]:
        st.subheader("Proposals / not agreed")
        st.caption("Ideas and suggestions that were raised but not agreed. These are not decisions.")
        for n, p in enumerate(record["proposals"], 1):
            st.markdown(f"**{n}. {p['proposal']}**  \nEvidence ({ids(p['evidence_segment_ids'])}): "
                        f"“{p['evidence_quote']}”")
        if not record["proposals"]:
            st.info("No open proposals.")

    with tabs[7]:
        st.subheader("Action items")
        st.caption("Only explicit assignments or commitments. Owner and deadline are shown only when "
                   "the recording states them.")
        if record["action_items"]:
            st.table([{
                "Task": a["task"],
                "Owner": a["owner"] or NOT_SPECIFIED,
                "Deadline": a["deadline"] or NOT_SPECIFIED,
                "Evidence": f"{ids(a['evidence_segment_ids'])}: “{a['evidence_quote']}”",
            } for a in record["action_items"]])
        else:
            st.info("No action items were assigned.")

    with tabs[8]:
        st.subheader("Withheld / rejected items")
        st.caption("Items the documentation model generated that failed the grounding checks. "
                   "They are not part of the meeting record.")
        for r in record["rejected_items"]:
            content = r["content"]
            title = (content.get("decision") or content.get("proposal") or content.get("task")
                     or content.get("topic") or content.get("summary") or "")
            with st.expander(f"{KIND_LABELS.get(r['kind'], r['kind'])}: {title[:90]}"):
                st.markdown("**Why it was withheld:**  \n" + "  \n".join(f"- {reason}" for reason in r["reasons"]))
                if content.get("evidence_quote"):
                    st.markdown(f"**Cited evidence** ({ids(content.get('evidence_segment_ids') or [])}): "
                                f"“{content['evidence_quote']}”")
                if "owner" in content or "deadline" in content:
                    st.markdown(f"**Attempted owner:** {content.get('owner') or NOT_SPECIFIED} · "
                                f"**Attempted deadline:** {content.get('deadline') or NOT_SPECIFIED}")
                st.json(content, expanded=False)
        if not record["rejected_items"]:
            st.info("Nothing was withheld: every generated item passed the grounding checks.")

with tabs[9]:
    st.subheader("Downloads")
    try:
        files = get(f"/jobs/{job_id}/exports").json()
    except (requests.RequestException, ValueError):
        files = []
        st.error("Could not load the export list from the backend.")
    if not files:
        st.info("Nothing to download yet.")
    for f in files:
        resp = get(f"/jobs/{job_id}/exports/{f['key']}", timeout=30)
        if resp.ok:
            st.download_button(f"⬇️ {f['label']} — {f['filename']}", data=resp.content,
                               file_name=f["filename"], mime=f["mime"], key=f"dl-{f['key']}",
                               on_click="ignore")
