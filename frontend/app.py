"""Streamlit UI. Run from repo root: streamlit run frontend/app.py"""

import os
import time

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
POLL_SECONDS = 2
UNSPECIFIED = "Unspecified"
STAGE_LABELS = {
    "validate": "Audio validation & normalisation",
    "transcribe": "Speech-to-text (Groq Whisper)",
    "refine": "Transcript refinement (Gemini LLM #1)",
    "document": "Meeting documentation (Gemini LLM #2)",
    "export": "Export",
}
STATUS_ICONS = {"pending": "⏳", "running": "🔄", "done": "✅", "failed": "❌", "skipped": "⏭️"}

st.set_page_config(page_title="Meeting Assistant", layout="wide")
st.title("Meeting Assistant")
st.caption("Speech-to-text → domain-aware refinement → meeting minutes, decisions and action items")


def ts(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def api_error(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail") or resp.text
    except ValueError:
        return resp.text or f"HTTP {resp.status_code}"


with st.sidebar:
    st.subheader("Backend")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        st.success(f"Connected ({BACKEND_URL})")
        for key, ok in health.get("config", {}).items():
            st.write(("✅ " if ok else "⚠️ ") + key)
    except requests.RequestException as exc:
        st.error(f"Backend unreachable at {BACKEND_URL}: {exc.__class__.__name__}")

uploaded = st.file_uploader(
    "Upload an English meeting recording",
    type=["wav", "mp3", "m4a", "ogg", "flac", "webm", "mp4"],
)

if st.button("Process recording", type="primary", disabled=uploaded is None):
    try:
        resp = requests.post(
            f"{BACKEND_URL}/jobs",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
            timeout=120,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc.__class__.__name__}")
    else:
        if resp.ok:
            st.session_state["job_id"] = resp.json()["id"]
        else:
            st.error(api_error(resp))


def show_status(job: dict) -> None:
    for stage in job["stages"]:
        icon = STATUS_ICONS.get(stage["status"], "")
        label = STAGE_LABELS.get(stage["name"], stage["name"])
        message = f" — {stage['message']}" if stage.get("message") else ""
        st.write(f"{icon} **{label}**{message}")


job_id = st.session_state.get("job_id")
if job_id:
    status_box = st.empty()
    job = None
    while True:
        try:
            resp = requests.get(f"{BACKEND_URL}/jobs/{job_id}", timeout=10)
        except requests.RequestException as exc:
            st.error(f"Lost contact with the backend: {exc.__class__.__name__}")
            break
        if not resp.ok:
            st.error(api_error(resp))
            break
        job = resp.json()
        with status_box.container():
            st.subheader(f"Processing: {job['filename']}")
            show_status(job)
        if job["status"] in ("done", "failed"):
            break
        time.sleep(POLL_SECONDS)

    if job and job["status"] == "failed":
        st.error(f"Processing failed: {job.get('error')}")

    if job:
        raw, refined, record = job.get("raw_transcript"), job.get("refined_transcript"), job.get("record")
        tabs = st.tabs(["Raw transcript", "Refined transcript", "Summary", "Minutes", "Decisions",
                        "Proposals / not agreed", "Action items", "Withheld items"])

        with tabs[0]:
            if raw:
                st.caption(f"Exactly as returned by {raw['stt_model']} — never modified.")
                for s in raw["segments"]:
                    st.markdown(f"`{ts(s['start'])}–{ts(s['end'])}` **#{s['id']}** {s['text']}")
            else:
                st.info("No raw transcript.")

        with tabs[1]:
            if refined:
                st.caption(f"Refined by {refined['refine_model']}. Edits that failed the safety "
                           "checks were rejected and the raw text kept.")
                rows = [{
                    "#": r["segment_id"],
                    "time": ts(r["start"]),
                    "raw": r["original_text"],
                    "refined": r["refined_text"],
                    "status": r["status"],
                    "rejected edit": r.get("rejected_text") or "",
                    "why rejected": "; ".join(r.get("issues") or []),
                } for r in refined["refinements"]]
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("No refined transcript.")

        if not record:
            for tab in tabs[2:]:
                with tab:
                    st.info("No meeting record.")
        else:
            with tabs[2]:
                st.write(record["summary"] or "_Summary withheld: it failed validation._")
                st.caption(f"Generated by {record['document_model']}")
            with tabs[3]:
                for m in record["minutes"] or []:
                    st.markdown(f"**{m['topic']}** — {m['discussion']}  \n"
                                f"_segments {', '.join('#' + str(i) for i in m['segment_ids'])}_")
                if not record["minutes"]:
                    st.info("No minutes.")
            with tabs[4]:
                for d in record["decisions"]:
                    st.markdown(f"- **{d['decision']}**  \n  _#{', #'.join(map(str, d['evidence_segment_ids']))}: "
                                f"\"{d['evidence_quote']}\"_")
                if not record["decisions"]:
                    st.info("No decisions were recorded.")
            with tabs[5]:
                for p in record["proposals"]:
                    st.markdown(f"- {p['proposal']}  \n  _#{', #'.join(map(str, p['evidence_segment_ids']))}: "
                                f"\"{p['evidence_quote']}\"_")
                if not record["proposals"]:
                    st.info("No open proposals.")
            with tabs[6]:
                if record["action_items"]:
                    st.dataframe([{
                        "task": a["task"],
                        "owner": a["owner"] or UNSPECIFIED,
                        "deadline": a["deadline"] or UNSPECIFIED,
                        "evidence": f"#{', #'.join(map(str, a['evidence_segment_ids']))}: \"{a['evidence_quote']}\"",
                    } for a in record["action_items"]], use_container_width=True, hide_index=True)
                else:
                    st.info("No action items were assigned.")
            with tabs[7]:
                st.caption("Generated items that failed grounding checks. They are not part of the record.")
                for r in record["rejected_items"]:
                    st.markdown(f"- **{r['kind']}**: {'; '.join(r['reasons'])}")
                    st.json(r["content"], expanded=False)
                if not record["rejected_items"]:
                    st.info("Nothing was withheld.")
